import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import time
from pathlib import Path
import argparse
from collections import defaultdict
from spline_cconv import ContinuousConv1D
from time_series import TimeSeries
from ema import EMA
from mmd import mmd
from tracker import Tracker
from vis import Visualizer
from gen_toy_data import gen_data
from utils import Rescaler, mkdir
from layers import Decoder, gan_loss
from toy_layers import (
    SeqGeneratorDiscrete,
    conv_ln_lrelu,
)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Encoder(nn.Module):
    def __init__(self, cconv, latent_size, norm_trans=True):
        super().__init__()
        self.cconv = cconv
        self.ls = nn.Sequential(
            nn.LeakyReLU(0.2),
            conv_ln_lrelu(64, 128),
            conv_ln_lrelu(128, 256),
            # conv_ln_lrelu(256, 512),
            # conv_ln_lrelu(512, 64),
            conv_ln_lrelu(256, 32),
        )
        conv_size = 416
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(conv_size, latent_size * 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(latent_size * 2, latent_size * 2)),
        )
        self.norm_trans = norm_trans
        if norm_trans:
            self.fc2 = nn.Sequential(
                spectral_norm(nn.Linear(latent_size, latent_size)),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Linear(latent_size, latent_size)),
            )

    def forward(self, cconv_graph, batch_size):
        x = self.cconv(*cconv_graph, batch_size)
        # expected shape: (batch_size, 448)
        x = self.ls(x).view(x.shape[0], -1)
        mu, logvar = self.fc(x).chunk(2, dim=1)
        if self.training:
            std = F.softplus(logvar)
            eps = torch.empty_like(std).normal_()
            # return mu + eps * std, mu, logvar, eps
            z = mu + eps * std
        else:
            z = mu
        if self.norm_trans:
            z = self.fc2(z)
        return z


class GridCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.ls = nn.Sequential(
            nn.LeakyReLU(0.2),
            conv_ln_lrelu(64, 128),
            conv_ln_lrelu(128, 256),
            # conv_ln_lrelu(256, 512),
            # conv_ln_lrelu(512, 1)
            conv_ln_lrelu(256, 1),
        )

    def forward(self, x):
        # expected shape: (batch_size, 448)
        return self.ls(x)


class ConvCritic(nn.Module):
    def __init__(self, cconv, latent_size, embed_size=13):
        super().__init__()
        self.cconv = cconv
        self.grid_critic = GridCritic()
        # self.x_dis = spectral_norm(nn.Linear(7, embed_size))

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
        x = x.squeeze(1)
        # x = self.x_dis(x)
        z = self.z_dis(z)
        xz = torch.cat((x, z), 1)
        xz = self.xz_dis(xz)
        x_out = self.x_linear(x).view(-1)
        xz_out = xz.view(-1)
        return xz_out, x_out


class PBiGAN(nn.Module):
    def __init__(self, encoder, decoder, ae_loss='mse'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ae_loss = ae_loss

    def forward(self, data, time, mask, cconv_graph, time_t, mask_t):
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

        return z_T, x_recon, z_gen, x_gen, ae_loss.mean()


def main():
    parser = argparse.ArgumentParser()

    default_dataset = 'toy-data.npz'
    parser.add_argument('--data', default=default_dataset)
    parser.add_argument('--seed', type=int, default=None)

    # training options
    parser.add_argument('--nz', type=int, default=32, help='latent size')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=8e-5,
                        help='generator learning rate')
    parser.add_argument('--dis-lr', type=float, default=1e-4,
                        help='discriminator learning rate')
    parser.add_argument('--min-lr', type=float, default=5e-5,
                        help='min learning rate for LR scheduler, '
                             '-1 to disable annealing')
    parser.add_argument('--min-dis-lr', type=float, default=7e-5,
                        help='min learning rate for discriminator '
                             'LR scheduler, -1 to disable annealing')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--overlap', type=float, default=.5,
                        help='kernel overlap')
    parser.add_argument('--no-norm-trans', action='store_true')

    # log options: 0 to disable plot-interval or save-interval
    parser.add_argument('--plot-interval', type=int, default=1)
    parser.add_argument('--save-interval', type=int, default=0)
    parser.add_argument('--prefix', default='pbigan')
    parser.add_argument('--comp', type=int, default=7,
                        help='continuous convolution kernel size')
    parser.add_argument('--ae', type=float, default=.1)
    parser.add_argument('--aeloss', default='smooth_l1',
                        help='autoencoding loss, options: mse, smooth_l1')

    parser.add_argument('--ema', dest='ema', type=int, default=-1,
                        help='start iteration of EMA, -1 to disable EMA')
    parser.add_argument('--ema-decay', type=float, default=.9999)
    parser.add_argument('--mmd', type=float, default=1)

    # squash is off when rescale is off
    parser.add_argument('--squash', dest='squash', action='store_const',
                        const=True, default=True,
                        help='bound generated time series value using tanh')
    parser.add_argument('--no-squash', dest='squash', action='store_const',
                        const=False)

    # rescale to [-1, 1]
    parser.add_argument('--rescale', dest='rescale', action='store_const',
                        const=True, default=True,
                        help='rescale the value of time series to [-1, 1]')
    parser.add_argument('--no-rescale', dest='rescale', action='store_const',
                        const=False)

    args = parser.parse_args()

    batch_size = args.batch_size
    nz = args.nz

    epochs = args.epoch
    plot_interval = args.plot_interval
    save_interval = args.save_interval

    try:
        npz = np.load(args.data)
        train_data = npz['data']
        train_time = npz['time']
        train_mask = npz['mask']
    except FileNotFoundError:
        if args.data != default_dataset:
            raise
        # Generate the default toy dataset from scratch
        train_data, train_time, train_mask, _, _ = gen_data(
            n_samples=10000, seq_len=200, max_time=1, poisson_rate=50,
            obs_span_rate=.25, save_file=default_dataset)

    _, in_channels, seq_len = train_data.shape
    train_time *= train_mask

    if args.seed is None:
        rnd = np.random.RandomState(None)
        random_seed = rnd.randint(np.iinfo(np.uint32).max)
    else:
        random_seed = args.seed
    rnd = np.random.RandomState(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Scale time
    max_time = 5
    train_time *= max_time

    squash = None
    rescaler = None
    if args.rescale:
        rescaler = Rescaler(train_data)
        train_data = rescaler.rescale(train_data)
        if args.squash:
            squash = torch.tanh

    out_channels = 64
    cconv_ref = 98

    train_dataset = TimeSeries(
        train_data, train_time, train_mask, label=None, max_time=max_time,
        cconv_ref=cconv_ref, overlap_rate=args.overlap, device=device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=train_dataset.collate_fn)
    n_train_batch = len(train_loader)

    time_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=train_dataset.collate_fn)

    test_loader = DataLoader(train_dataset, batch_size=batch_size,
                             collate_fn=train_dataset.collate_fn)

    grid_decoder = SeqGeneratorDiscrete(in_channels, nz, squash)
    decoder = Decoder(grid_decoder, max_time=max_time).to(device)

    cconv = ContinuousConv1D(
        in_channels, out_channels, max_time, cconv_ref,
        overlap_rate=args.overlap, kernel_size=args.comp, norm=True).to(device)
    encoder = Encoder(cconv, nz, not args.no_norm_trans).to(device)

    pbigan = PBiGAN(encoder, decoder, args.aeloss).to(device)

    critic_cconv = ContinuousConv1D(
        in_channels, out_channels, max_time, cconv_ref,
        overlap_rate=args.overlap, kernel_size=args.comp, norm=True).to(device)
    critic = ConvCritic(critic_cconv, nz).to(device)

    ema = None
    if args.ema >= 0:
        ema = EMA(pbigan, args.ema_decay, args.ema)

    optimizer = optim.Adam(
        pbigan.parameters(), lr=args.lr, weight_decay=args.wd)
    critic_optimizer = optim.Adam(
        critic.parameters(), lr=args.dis_lr, weight_decay=args.wd)

    scheduler = None
    if args.min_lr > 0:
        lr_steps = 10
        step_size = epochs // lr_steps
        gamma = (args.min_lr / args.lr)**(1 / lr_steps)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

    dis_scheduler = None
    if args.min_dis_lr > 0:
        lr_steps = 10
        step_size = epochs // lr_steps
        gamma = (args.min_dis_lr / args.dis_lr)**(1 / lr_steps)
        dis_scheduler = optim.lr_scheduler.StepLR(
            critic_optimizer, step_size=step_size, gamma=gamma)

    path = '{}_{}'.format(
        args.prefix, datetime.now().strftime('%m%d.%H%M%S'))

    output_dir = Path('results') / 'toy-pbigan' / path
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

    tracker = Tracker(log_dir, n_train_batch)
    visualizer = Visualizer(encoder, decoder, batch_size, max_time,
                            test_loader, rescaler, output_dir, device)
    start = time.time()
    epoch_start = start

    for epoch in range(start_epoch, epochs):
        loss_breakdown = defaultdict(float)

        for ((val, idx, mask, _, cconv_graph),
             (_, idx_t, mask_t, index, _)) in zip(
                 train_loader, time_loader):

            z_enc, x_recon, z_gen, x_gen, ae_loss = pbigan(
                val, idx, mask, cconv_graph, idx_t, mask_t)

            cconv_graph_gen = train_dataset.make_graph(
                x_gen, idx_t, mask_t, index)

            real = critic(cconv_graph, batch_size, z_enc)
            fake = critic(cconv_graph_gen, batch_size, z_gen)

            D_loss = gan_loss(real, fake, 1, 0)

            critic_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            critic_optimizer.step()

            G_loss = gan_loss(real, fake, 0, 1)

            mmd_loss = mmd(z_enc, z_gen)

            loss = G_loss + ae_loss * args.ae + mmd_loss * args.mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema:
                ema.update()

            loss_breakdown['D'] += D_loss.item()
            loss_breakdown['G'] += G_loss.item()
            loss_breakdown['AE'] += ae_loss.item()
            loss_breakdown['MMD'] += mmd_loss.item()
            loss_breakdown['total'] += loss.item()

        if scheduler:
            scheduler.step()
        if dis_scheduler:
            dis_scheduler.step()

        cur_time = time.time()
        tracker.log(
            epoch, loss_breakdown, cur_time - epoch_start, cur_time - start)

        if plot_interval > 0 and (epoch + 1) % plot_interval == 0:
            if ema:
                ema.apply()
                visualizer.plot(epoch)
                ema.restore()
            else:
                visualizer.plot(epoch)

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
