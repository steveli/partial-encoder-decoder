import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import time
from pathlib import Path
import argparse
from collections import defaultdict
from spline_cconv import ContinuousConv1D
import time_series
from ema import EMA
from utils import count_parameters, mkdir, make_scheduler
from mmd import mmd
from tracker import Tracker
from evaluate import Evaluator
from sn_layers import (
    InvertibleLinearResNet,
    Classifier,
    GridDecoder,
    GridEncoder,
    Decoder,
    gan_loss,
)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Encoder(nn.Module):
    def __init__(self, cconv, latent_size, channels, trans_layers=1):
        super().__init__()
        self.cconv = cconv
        self.grid_encoder = GridEncoder(channels, latent_size * 2)
        self.trans = InvertibleLinearResNet(
            latent_size, latent_size, trans_layers).to(device)

    def forward(self, cconv_graph, batch_size):
        x = self.cconv(*cconv_graph, batch_size)
        mu, logvar = self.grid_encoder(x).chunk(2, dim=1)
        if self.training:
            std = F.softplus(logvar)
            qz_x = Normal(mu, std)
            z_0 = qz_x.rsample()
            z_T = self.trans(z_0)
        else:
            z_T = self.trans(mu)
        return z_T


class ConvCritic(nn.Module):
    def __init__(self, cconv, latent_size, channels, embed_size=32):
        super().__init__()

        self.cconv = cconv
        self.grid_critic = GridEncoder(channels, embed_size)

        self.z_dis = nn.Sequential(
            spectral_norm(nn.Linear(latent_size, embed_size)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(embed_size, embed_size)),
        )

        self.x_linear = spectral_norm(nn.Linear(embed_size, 1))

        self.xz_dis = nn.Sequential(
            spectral_norm(nn.Linear(embed_size * 2, embed_size)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(embed_size, 1)),
        )

    def forward(self, cconv_graph, batch_size, z):
        x = self.cconv(*cconv_graph, batch_size)
        x = self.grid_critic(x)
        z = self.z_dis(z)
        xz = torch.cat((x, z), 1)
        xz = self.xz_dis(xz)
        x_out = self.x_linear(x).view(-1)
        xz_out = xz.view(-1)
        return xz_out, x_out


class PBiGAN(nn.Module):
    def __init__(self, encoder, decoder, classifier, ae_loss='mse'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.ae_loss = ae_loss

    def forward(self, data, time, mask, y, cconv_graph, time_t, mask_t):
        batch_size = len(data)
        z_T = self.encoder(cconv_graph, batch_size)

        z_gen = torch.empty_like(z_T).normal_()
        x_gen = self.decoder(z_gen, time_t, mask_t)

        x_recon = self.decoder(z_T, time, mask)

        if self.ae_loss == 'mse':
            ae_loss = F.mse_loss(x_recon, data, reduction='none') * mask
        elif self.ae_loss == 'smooth_l1':
            ae_loss = F.smooth_l1_loss(x_recon, data, reduction='none') * mask

        ae_loss = ae_loss.sum((-1, -2))

        y_logit = self.classifier(z_T).view(-1)

        # cls_loss: -log p(y|z)
        cls_loss = F.binary_cross_entropy_with_logits(
            y_logit, y.expand_as(y_logit), reduction='none')

        return z_T, x_recon, z_gen, x_gen, ae_loss.mean(), cls_loss.mean()

    def predict(self, data, time, mask, cconv_graph):
        batch_size = len(data)
        z_T = self.encoder(cconv_graph, batch_size)
        y_logit = self.classifier(z_T).view(-1)
        return y_logit


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='mimic3.npz',
                        help='data file')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed. Randomly set if not specified.')

    # training options
    parser.add_argument('--nz', type=int, default=32,
                        help='dimension of latent variable')
    parser.add_argument('--epoch', type=int, default=500,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    # Use smaller test batch size to accommodate more importance samples
    parser.add_argument('--test-batch-size', type=int, default=32,
                        help='batch size for validation and test set')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='encoder/decoder learning rate')
    parser.add_argument('--dis-lr', type=float, default=3e-4,
                        help='discriminator learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-4,
                        help='min encoder/decoder learning rate for LR '
                             'scheduler. -1 to disable annealing')
    parser.add_argument('--min-dis-lr', type=float, default=1.5e-4,
                        help='min discriminator learning rate for LR '
                             'scheduler. -1 to disable annealing')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--overlap', type=float, default=.5,
                        help='kernel overlap')
    parser.add_argument('--cls', type=float, default=1,
                        help='classification weight')
    parser.add_argument('--clsdep', type=int, default=1,
                        help='number of layers for classifier')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='AUC evaluation interval. '
                             '0 to disable evaluation.')
    parser.add_argument('--save-interval', type=int, default=0,
                        help='interval to save models. 0 to disable saving.')
    parser.add_argument('--prefix', default='pbigan',
                        help='prefix of output directory')
    parser.add_argument('--comp', type=int, default=7,
                        help='continuous convolution kernel size')
    parser.add_argument('--ae', type=float, default=1,
                        help='autoencoding regularization strength')
    parser.add_argument('--aeloss', default='mse',
                        help='autoencoding loss. (options: mse, smooth_l1)')
    parser.add_argument('--dec-ch', default='8-16-16',
                        help='decoder architecture')
    parser.add_argument('--enc-ch', default='64-32-32-16',
                        help='encoder architecture')
    parser.add_argument('--dis-ch', default=None,
                        help='discriminator architecture. Use encoder '
                             'architecture if unspecified.')
    parser.add_argument('--rescale', dest='rescale', action='store_const',
                        const=True, default=True,
                        help='if set, rescale time to [-1, 1]')
    parser.add_argument('--no-rescale', dest='rescale', action='store_const',
                        const=False)
    parser.add_argument('--cconvnorm', dest='cconv_norm',
                        action='store_const', const=True, default=True,
                        help='if set, normalize continuous convolutional '
                             'layer using mean pooling')
    parser.add_argument('--no-cconvnorm', dest='cconv_norm',
                        action='store_const', const=False)
    parser.add_argument('--cconv-ref', type=int, default=98,
                        help='number of evenly-spaced reference locations '
                             'for continuous convolutional layer')
    parser.add_argument('--dec-ref', type=int, default=128,
                        help='number of evenly-spaced reference locations '
                             'for decoder')
    parser.add_argument('--trans', type=int, default=2,
                        help='number of encoder layers')
    parser.add_argument('--ema', dest='ema', type=int, default=0,
                        help='start epoch of exponential moving average '
                             '(EMA). -1 to disable EMA')
    parser.add_argument('--ema-decay', type=float, default=.9999,
                        help='EMA decay')
    parser.add_argument('--mmd', type=float, default=1,
                        help='MMD strength for latent variable')

    args = parser.parse_args()

    nz = args.nz

    epochs = args.epoch
    eval_interval = args.eval_interval
    save_interval = args.save_interval

    if args.seed is None:
        rnd = np.random.RandomState(None)
        random_seed = rnd.randint(np.iinfo(np.uint32).max)
    else:
        random_seed = args.seed
    rnd = np.random.RandomState(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    max_time = 5
    cconv_ref = args.cconv_ref
    overlap = args.overlap
    train_dataset, val_dataset, test_dataset = time_series.split_data(
        args.data, rnd, max_time, cconv_ref, overlap, device, args.rescale)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, collate_fn=train_dataset.collate_fn)
    n_train_batch = len(train_loader)

    time_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(
        val_dataset, batch_size=args.test_batch_size, shuffle=False,
        collate_fn=val_dataset.collate_fn)

    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        collate_fn=test_dataset.collate_fn)

    in_channels, seq_len = train_dataset.data.shape[1:]

    if args.dis_ch is None:
        args.dis_ch = args.enc_ch

    dec_channels = [int(c) for c in args.dec_ch.split('-')] + [in_channels]
    enc_channels = [int(c) for c in args.enc_ch.split('-')]
    dis_channels = [int(c) for c in args.dis_ch.split('-')]

    out_channels = enc_channels[0]

    squash = torch.sigmoid
    if args.rescale:
        squash = torch.tanh

    dec_ch_up = 2**(len(dec_channels) - 2)
    assert args.dec_ref % dec_ch_up == 0, (
        f'--dec-ref={args.dec_ref} is not divided by {dec_ch_up}.')
    dec_len0 = args.dec_ref // dec_ch_up
    grid_decoder = GridDecoder(nz, dec_channels, dec_len0, squash)

    decoder = Decoder(
        grid_decoder, max_time=max_time, dec_ref=args.dec_ref).to(device)
    cconv = ContinuousConv1D(
        in_channels, out_channels, max_time, cconv_ref, overlap_rate=overlap,
        kernel_size=args.comp, norm=args.cconv_norm).to(device)
    encoder = Encoder(cconv, nz, enc_channels, args.trans).to(device)

    classifier = Classifier(nz, args.clsdep).to(device)

    pbigan = PBiGAN(
        encoder, decoder, classifier, ae_loss=args.aeloss).to(device)

    ema = None
    if args.ema >= 0:
        ema = EMA(pbigan, args.ema_decay, args.ema)

    critic_cconv = ContinuousConv1D(
        in_channels, out_channels, max_time, cconv_ref, overlap_rate=overlap,
        kernel_size=args.comp, norm=args.cconv_norm).to(device)
    critic_embed = 32
    critic = ConvCritic(
        critic_cconv, nz, dis_channels, critic_embed).to(device)

    optimizer = optim.Adam(
        pbigan.parameters(), lr=args.lr,
        betas=(0, .999), weight_decay=args.wd)
    critic_optimizer = optim.Adam(
        critic.parameters(), lr=args.dis_lr,
        betas=(0, .999), weight_decay=args.wd)

    scheduler = make_scheduler(optimizer, args.lr, args.min_lr, epochs)
    dis_scheduler = make_scheduler(
        critic_optimizer, args.dis_lr, args.min_dis_lr, epochs)

    path = '{}_{}'.format(
        args.prefix, datetime.now().strftime('%m%d.%H%M%S'))

    output_dir = Path('results') / 'mimic3-pbigan' / path
    print(output_dir)
    log_dir = mkdir(output_dir / 'log')
    model_dir = mkdir(output_dir / 'model')

    start_epoch = 0

    with (log_dir / 'seed.txt').open('w') as f:
        print(random_seed, file=f)
    with (log_dir / 'gpu.txt').open('a') as f:
        print(torch.cuda.device_count(), start_epoch, file=f)
    with (log_dir / 'args.txt').open('w') as f:
        for key, val in sorted(vars(args).items()):
            print(f'{key}: {val}', file=f)
    with (log_dir / 'params.txt').open('w') as f:
        def print_params_count(module, name):
            try:   # sum counts if module is a list
                params_count = sum(count_parameters(m) for m in module)
            except TypeError:
                params_count = count_parameters(module)
            print(f'{name} {params_count}', file=f)
        print_params_count(grid_decoder, 'grid_decoder')
        print_params_count(decoder, 'decoder')
        print_params_count(cconv, 'cconv')
        print_params_count(encoder, 'encoder')
        print_params_count(classifier, 'classifier')
        print_params_count(pbigan, 'pbigan')
        print_params_count(critic, 'critic')
        print_params_count([pbigan, critic], 'total')

    tracker = Tracker(log_dir, n_train_batch)
    evaluator = Evaluator(pbigan, val_loader, test_loader, log_dir)
    start = time.time()
    epoch_start = start

    batch_size = args.batch_size

    for epoch in range(start_epoch, epochs):
        loss_breakdown = defaultdict(float)
        epoch_start = time.time()

        if epoch >= 40:
            args.cls = 200

        for ((val, idx, mask, y, _, cconv_graph),
             (_, idx_t, mask_t, _, index, _)) in zip(
                 train_loader, time_loader):

            z_enc, x_recon, z_gen, x_gen, ae_loss, cls_loss = pbigan(
                val, idx, mask, y, cconv_graph, idx_t, mask_t)

            cconv_graph_gen = train_dataset.make_graph(
                x_gen, idx_t, mask_t, index)

            # Don't need pbigan.requires_grad_(False);
            # critic takes as input only the detached tensors.
            real = critic(cconv_graph, batch_size, z_enc.detach())
            detached_graph = [[cat_y.detach() for cat_y in x] if i == 2 else x
                              for i, x in enumerate(cconv_graph_gen)]
            fake = critic(detached_graph, batch_size, z_gen.detach())

            D_loss = gan_loss(real, fake, 1, 0)

            critic_optimizer.zero_grad()
            D_loss.backward()
            critic_optimizer.step()

            for p in critic.parameters():
                p.requires_grad_(False)
            real = critic(cconv_graph, batch_size, z_enc)
            fake = critic(cconv_graph_gen, batch_size, z_gen)

            G_loss = gan_loss(real, fake, 0, 1)

            mmd_loss = mmd(z_enc, z_gen)

            loss = (G_loss + ae_loss * args.ae + cls_loss * args.cls
                    + mmd_loss * args.mmd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for p in critic.parameters():
                p.requires_grad_(True)

            if ema:
                ema.update()

            loss_breakdown['D'] += D_loss.item()
            loss_breakdown['G'] += G_loss.item()
            loss_breakdown['AE'] += ae_loss.item()
            loss_breakdown['MMD'] += mmd_loss.item()
            loss_breakdown['CLS'] += cls_loss.item()
            loss_breakdown['total'] += loss.item()

        if scheduler:
            scheduler.step()
        if dis_scheduler:
            dis_scheduler.step()

        cur_time = time.time()
        tracker.log(
            epoch, loss_breakdown, cur_time - epoch_start, cur_time - start)

        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            if ema:
                ema.apply()
                evaluator.evaluate(epoch)
                ema.restore()
            else:
                evaluator.evaluate(epoch)

        model_dict = {
            'pbigan': pbigan.state_dict(),
            'critic': critic.state_dict(),
            'ema': ema.state_dict() if ema else None,
            'epoch': epoch + 1,
            'args': args,
        }
        torch.save(model_dict, str(log_dir / 'model.pth'))
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            torch.save(model_dict, str(model_dir / f'{epoch:04d}.pth'))

    print(output_dir)


if __name__ == '__main__':
    main()
