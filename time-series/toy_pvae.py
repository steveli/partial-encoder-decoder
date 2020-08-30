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
from tracker import Tracker
from vis import Visualizer
from gen_toy_data import gen_data
from utils import Rescaler, mkdir, make_scheduler
from layers import Decoder
from toy_layers import SeqGeneratorDiscrete, conv_ln_lrelu


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Encoder(nn.Module):
    def __init__(self, latent_size, cconv):
        super().__init__()
        self.cconv = cconv
        self.ls = nn.Sequential(
            nn.LeakyReLU(0.2),
            conv_ln_lrelu(64, 128),
            conv_ln_lrelu(128, 256),
            conv_ln_lrelu(256, 512),
            conv_ln_lrelu(512, 64),
        )
        conv_size = 448
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(conv_size, latent_size * 2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(latent_size * 2, latent_size * 2)),
        )

    def forward(self, cconv_graph, batch_size):
        x = self.cconv(*cconv_graph, batch_size)
        # expected shape: (batch_size, 448)
        x = self.ls(x).view(x.shape[0], -1)
        mu, logvar = self.fc(x).chunk(2, dim=1)
        std = torch.exp(logvar * .5)
        eps = torch.empty_like(std).normal_()
        return mu + eps * std, mu, logvar, eps


class PVAE(nn.Module):
    def __init__(self, encoder, decoder, sigma=.2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sigma = sigma

    def forward(self, data, time, mask, cconv_graph):
        batch_size = len(data)
        z, mu, logvar, eps = self.encoder(cconv_graph, batch_size)
        x_recon = self.decoder(z, time, mask)
        # Gaussian noise
        recon_loss = (1 / (2 * self.sigma**2) * F.mse_loss(
            x_recon * mask, data * mask, reduction='none') * mask).sum((1, 2))
        kl_loss = .5 * (z**2 - logvar - eps**2).sum(1)
        loss = recon_loss.mean() + kl_loss.mean()
        return loss


def main():
    parser = argparse.ArgumentParser()

    default_dataset = 'toy-data.npz'
    parser.add_argument('--data', default=default_dataset,
                        help='data file')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed. Randomly set if not specified.')

    # training options
    parser.add_argument('--nz', type=int, default=32,
                        help='dimension of latent variable')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--min-lr', type=float, default=-1,
                        help='min learning rate for LR scheduler. '
                             '-1 to disable annealing')
    parser.add_argument('--plot-interval', type=int, default=10,
                        help='plot interval. 0 to disable plotting.')
    parser.add_argument('--save-interval', type=int, default=0,
                        help='interval to save models. 0 to disable saving.')
    parser.add_argument('--prefix', default='pvae',
                        help='prefix of output directory')
    parser.add_argument('--comp', type=int, default=5,
                        help='continuous convolution kernel size')
    parser.add_argument('--sigma', type=float, default=.2,
                        help='standard deviation for Gaussian likelihood')
    parser.add_argument('--overlap', type=float, default=.5,
                        help='kernel overlap')
    # squash is off when rescale is off
    parser.add_argument('--squash', dest='squash', action='store_const',
                        const=True, default=True,
                        help='bound the generated time series value '
                             'using tanh')
    parser.add_argument('--no-squash', dest='squash', action='store_const',
                        const=False)

    # rescale to [-1, 1]
    parser.add_argument('--rescale', dest='rescale', action='store_const',
                        const=True, default=True,
                        help='if set, rescale time to [-1, 1]')
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

    test_batch_size = 64
    test_loader = DataLoader(train_dataset, batch_size=test_batch_size,
                             collate_fn=train_dataset.collate_fn)

    grid_decoder = SeqGeneratorDiscrete(in_channels, nz, squash)
    decoder = Decoder(grid_decoder, max_time=max_time).to(device)

    cconv = ContinuousConv1D(
        in_channels, out_channels, max_time, cconv_ref,
        overlap_rate=args.overlap, kernel_size=args.comp, norm=True).to(device)

    encoder = Encoder(nz, cconv).to(device)

    pvae = PVAE(
        encoder, decoder, sigma=args.sigma).to(device)

    optimizer = optim.Adam(pvae.parameters(), lr=args.lr)

    scheduler = make_scheduler(optimizer, args.lr, args.min_lr, epochs)

    path = '{}_{}_{}'.format(
        args.prefix, datetime.now().strftime('%m%d.%H%M%S'),
        '_'.join([f'lr_{args.lr:g}']))

    output_dir = Path('results') / 'toy-pvae' / path
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
    visualizer = Visualizer(encoder, decoder, test_batch_size, max_time,
                            test_loader, rescaler, output_dir, device)
    start = time.time()
    epoch_start = start

    for epoch in range(start_epoch, epochs):
        loss_breakdown = defaultdict(float)
        for val, idx, mask, _, cconv_graph in train_loader:
            optimizer.zero_grad()
            loss = pvae(val, idx, mask, cconv_graph)
            loss.backward()
            optimizer.step()
            loss_breakdown['loss'] += loss.item()

        if scheduler:
            scheduler.step()

        cur_time = time.time()
        tracker.log(
            epoch, loss_breakdown, cur_time - epoch_start, cur_time - start)

        if plot_interval > 0 and (epoch + 1) % plot_interval == 0:
            visualizer.plot(epoch)

        model_dict = {
            'pvae': pvae.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        torch.save(model_dict, str(log_dir / 'model.pth'))
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            torch.save(model_dict, str(model_dir / f'{epoch:04d}.pth'))

    print(output_dir)


if __name__ == '__main__':
    main()
