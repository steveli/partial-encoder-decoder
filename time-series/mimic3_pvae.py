import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import time
from pathlib import Path
import argparse
import math
from collections import defaultdict
from spline_cconv import ContinuousConv1D
import time_series
import flow
from ema import EMA
from utils import count_parameters, mkdir
from tracker import Tracker
from evaluate import Evaluator
from layers import (
    Classifier,
    GridDecoder,
    GridEncoder,
    Decoder,
)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Encoder(nn.Module):
    def __init__(self, cconv, latent_size, channels, flow_depth=2):
        super().__init__()
        self.cconv = cconv

        if flow_depth > 0:
            hidden_size = latent_size * 2
            flow_layers = [flow.InverseAutoregressiveFlow(
                latent_size, hidden_size, latent_size)
                for _ in range(flow_depth)]

            flow_layers.append(flow.Reverse(latent_size))
            self.q_z_flow = flow.FlowSequential(*flow_layers)
            self.enc_chunk = 3
        else:
            self.q_z_flow = None
            self.enc_chunk = 2

        self.grid_encoder = GridEncoder(channels, latent_size * self.enc_chunk)

    def forward(self, cconv_graph, batch_size, iw_samples=3):
        x = self.cconv(*cconv_graph, batch_size)
        grid_enc = self.grid_encoder(x).chunk(self.enc_chunk, dim=1)
        mu, logvar = grid_enc[:2]
        std = F.softplus(logvar)
        qz_x = Normal(mu, std)
        z_0 = qz_x.rsample([iw_samples])
        log_q_z_0 = qz_x.log_prob(z_0)
        if self.q_z_flow:
            z_T, log_q_z_flow = self.q_z_flow(z_0, context=grid_enc[2])
            log_q_z = (log_q_z_0 + log_q_z_flow).sum(-1)
        else:
            z_T, log_q_z = z_0, log_q_z_0.sum(-1)
        return z_T, log_q_z


def masked_loss(loss_fn, pred, data, mask):
    # return (loss_fn(pred * mask, data * mask,
    #                 reduction='none') * mask).mean()
    # Expand data shape from (batch_size, d) to (iw_samples, batch_size, d)
    return loss_fn(pred, data.expand_as(pred), reduction='none') * mask


class PVAE(nn.Module):
    def __init__(self, encoder, decoder, classifier, sigma=.2, cls_weight=100):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.sigma = sigma
        self.cls_weight = cls_weight

    def forward(self, data, time, mask, y, cconv_graph, iw_samples=3,
                ts_lambda=1, kl_lambda=1):
        batch_size = len(data)
        z_T, log_q_z = self.encoder(cconv_graph, batch_size, iw_samples)

        pz = Normal(torch.zeros_like(z_T), torch.ones_like(z_T))
        log_p_z = pz.log_prob(z_T).sum(-1)
        # kl_loss: log q(z|x) - log p(z)
        kl_loss = log_q_z - log_p_z

        var2 = 2 * self.sigma**2
        # half_log2pivar: log(2 * pi * sigma^2) / 2
        half_log2pivar = .5 * math.log(math.pi * var2)

        # Multivariate Gaussian log-likelihood:
        # -D/2 * log(2*pi*sigma^2) - 1/2 \sum_{i=1}^D (x_i - mu_i)^2 / sigma^2
        def neg_gaussian_logp(pred, data, mask=None):
            se = F.mse_loss(pred, data.expand_as(pred), reduction='none')
            if mask is None:
                return se / var2 + half_log2pivar
            return (se / var2 + half_log2pivar) * mask

        # Reshape z to accommodate modules with strict input shape
        # requirements such as convolutional layers.
        # Expected shape of x_recon: (iw_samples * batch_size, C, L)
        z_flat = z_T.view(-1, *z_T.shape[2:])
        x_recon = self.decoder(
            z_flat,
            time.repeat((iw_samples, 1, 1)),
            mask.repeat((iw_samples, 1, 1)))

        # Gaussian noise for time series
        # data shape :(batch_size, C, L)
        # x_recon shape: (iw_samples * batch_size, C, L)
        x_recon = x_recon.view(iw_samples, *data.shape)
        neg_logp = neg_gaussian_logp(x_recon, data, mask)
        # neg_logp: -log p(x|z)
        neg_logp = neg_logp.sum((-1, -2))

        y_logit = self.classifier(z_flat).view(iw_samples, -1)

        # cls_loss: -log p(y|z)
        cls_loss = F.binary_cross_entropy_with_logits(
            y_logit, y.expand_as(y_logit), reduction='none')

        # elbo_x = log p(x|z) + log p(z) - log q(z|x)
        elbo_x = -(neg_logp * ts_lambda + kl_loss * kl_lambda)

        with torch.no_grad():
            is_weight = F.softmax(elbo_x, 0)

        # IWAE loss: -log E[p(x|z) p(z) / q(z|x)]
        # Here we ignore the constant shift of -log(k_samples)
        loss_x = -elbo_x.logsumexp(0).mean()
        loss_y = (is_weight * cls_loss).sum(0).mean()
        loss = loss_x + loss_y * self.cls_weight

        # For debugging
        x_se = masked_loss(F.mse_loss, x_recon, data, mask)
        mse = x_se.sum((-1, -2)) / mask.sum((-1, -2)).clamp(min=1)

        CE = (is_weight * cls_loss).sum(0).mean().item()
        loss_breakdown = {
            'loss': loss.item(),
            'reconst.': neg_logp.mean().item() * ts_lambda,
            'MSE': mse.mean().item(),
            'KL': kl_loss.mean().item() * kl_lambda,
            'CE': CE,
            'classif.': CE * self.cls_weight,
        }
        return loss, z_T, elbo_x, loss_breakdown

    def predict(self, data, time, mask, cconv_graph, iw_samples=50):
        dummy_y = data.new_zeros(len(data))
        _, z, elbo, _ = self(
            data, time, mask, dummy_y, cconv_graph, iw_samples)
        z_flat = z.view(-1, *z.shape[2:])
        pred_logit = self.classifier(z_flat).view(iw_samples, -1)
        is_weight = F.softmax(elbo, 0)

        # Importance reweighted predictive probability
        # p(y|x) =~ E_{q_IW(z|x)}[p(y|z)]
        py_z = torch.sigmoid(pred_logit)
        expected_py_z = (is_weight * py_z).sum(0)
        return expected_py_z


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='mimic3-rescaled.npz')
    parser.add_argument('--seed', type=int, default=None)

    # training options
    parser.add_argument('--nz', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    # Use smaller test batch size to accommodate more importance samples
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--train-k', type=int, default=8)
    parser.add_argument('--test-k', type=int, default=50)
    parser.add_argument('--flow', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--enc-lr', type=float, default=1e-4)
    parser.add_argument('--dec-lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=None)
    parser.add_argument('--wd', type=float, default=1e-3)   # weight decay
    parser.add_argument('--overlap', type=float, default=.5)   # kernel overlap
    parser.add_argument('--cls', type=float, default=200)
    parser.add_argument('--clsdep', type=int, default=1)
    parser.add_argument('--ts', type=float, default=1)
    parser.add_argument('--kl', type=float, default=.1)

    # log options: 0 to disable plot-interval or save-interval
    parser.add_argument('--plot-interval', type=int, default=1)
    parser.add_argument('--save-interval', type=int, default=0)
    parser.add_argument('--prefix', default='pvae')
    parser.add_argument('--comp', type=int, default=7)
    parser.add_argument('--sigma', type=float, default=.2)
    parser.add_argument('--dec-ch', default='8-16-16')
    parser.add_argument('--enc-ch', default='64-32-32-16')
    # rescale to [-1, 1]
    parser.add_argument('--rescale', dest='rescale', action='store_const',
                        const=True, default=True)
    parser.add_argument('--no-rescale', dest='rescale', action='store_const',
                        const=False)
    parser.add_argument('--cconvnorm', dest='cconv_norm',
                        action='store_const', const=True, default=True)
    parser.add_argument('--no-cconvnorm', dest='cconv_norm',
                        action='store_const', const=False)
    parser.add_argument('--cconv-ref', type=int, default=98)
    parser.add_argument('--dec-ref', type=int, default=128)
    # ema: -1 to disable EMA, otherwise ema specifies the start_iter of EMA
    parser.add_argument('--ema', dest='ema', type=int, default=0)
    parser.add_argument('--ema-decay', type=float, default=.9999)

    args = parser.parse_args()

    nz = args.nz

    epochs = args.epoch
    plot_interval = args.plot_interval
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

    val_loader = DataLoader(
        val_dataset, batch_size=args.test_batch_size, shuffle=False,
        collate_fn=val_dataset.collate_fn)

    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        collate_fn=test_dataset.collate_fn)

    in_channels, seq_len = train_dataset.data.shape[1:]

    dec_channels = [int(c) for c in args.dec_ch.split('-')] + [in_channels]
    enc_channels = [int(c) for c in args.enc_ch.split('-')]

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

    cconv = ContinuousConv1D(in_channels, out_channels, max_time, cconv_ref,
                             overlap_rate=overlap, kernel_size=args.comp,
                             norm=args.cconv_norm).to(device)
    encoder = Encoder(cconv, nz, enc_channels, args.flow).to(device)

    classifier = Classifier(nz, args.clsdep).to(device)

    pvae = PVAE(
        encoder, decoder, classifier, args.sigma, args.cls).to(device)

    ema = None
    if args.ema >= 0:
        ema = EMA(pvae, args.ema_decay, args.ema)

    other_params = [param for name, param in pvae.named_parameters()
                    if not (name.startswith('decoder.grid_decoder')
                            or name.startswith('encoder.grid_encoder'))]
    params = [
        {'params': decoder.grid_decoder.parameters(), 'lr': args.dec_lr},
        {'params': encoder.grid_encoder.parameters(), 'lr': args.enc_lr},
        {'params': other_params},
    ]

    optimizer = optim.Adam(
        params, lr=args.lr, weight_decay=args.wd)

    scheduler = None
    if args.min_lr is not None and args.min_lr != args.lr:
        lr_steps = 10
        step_size = epochs // lr_steps
        gamma = (args.min_lr / args.lr)**(1 / lr_steps)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

    path = '{}_{}'.format(
        args.prefix, datetime.now().strftime('%m%d.%H%M%S'))

    output_dir = Path('results') / 'mimic3-pvae' / path
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
        print_params_count(pvae, 'pvae')
        print_params_count(pvae, 'total')

    tracker = Tracker(log_dir, n_train_batch)
    evaluator = Evaluator(pvae, val_loader, test_loader, log_dir,
                          eval_args={'iw_samples': args.test_k})
    start = time.time()
    epoch_start = start

    for epoch in range(start_epoch, epochs):
        loss_breakdown = defaultdict(float)
        epoch_start = time.time()
        for (val, idx, mask, y, _, cconv_graph) in train_loader:
            optimizer.zero_grad()
            loss, _, _, loss_info = pvae(
                val, idx, mask, y, cconv_graph, args.train_k, args.ts, args.kl)
            loss.backward()
            optimizer.step()

            if ema:
                ema.update()

            for loss_name, loss_val in loss_info.items():
                loss_breakdown[loss_name] += loss_val

        if scheduler:
            scheduler.step()

        cur_time = time.time()
        tracker.log(
            epoch, loss_breakdown, cur_time - epoch_start, cur_time - start)

        if plot_interval > 0 and (epoch + 1) % plot_interval == 0:
            if ema:
                ema.apply()
                evaluator.evaluate(epoch)
                ema.restore()
            else:
                evaluator.evaluate(epoch)

        model_dict = {
            'pvae': pvae.state_dict(),
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
