from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
import math
import sys
import logging
from pathlib import Path
from datetime import datetime
import pprint
import argparse
from masked_celeba import BlockMaskedCelebA, IndepMaskedCelebA
from celeba_decoder import ConvDecoder
from celeba_encoder import ConvEncoder
from utils import mkdir
from visualize import Visualizer


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class PVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask, k_samples=5, kl_weight=1, ae_weight=1):
        z_T, log_q_z = self.encoder(x * mask, k_samples)

        pz = Normal(torch.zeros_like(z_T), torch.ones_like(z_T))
        log_p_z = pz.log_prob(z_T).sum(-1)
        # kl_loss: log q(z|x) - log p(z)
        kl_loss = log_q_z - log_p_z

        # Reshape z to accommodate modules with strict input shape requirements
        # such as convolutional layers.
        x_logit, x_recon = self.decoder(z_T.view(-1, *z_T.shape[2:]))
        expanded_mask = mask[None]
        masked_logit = x_logit.view(k_samples, *x.shape) * expanded_mask
        masked_x = (x * mask)[None].expand_as(masked_logit)
        # Bernoulli noise
        bce = F.binary_cross_entropy_with_logits(
            masked_logit, masked_x, reduction='none')
        recon_loss = (bce * expanded_mask).sum((2, 3, 4))

        # elbo = log p(x|z) + log p(z) - log q(z|x)
        elbo = -(recon_loss * ae_weight + kl_loss * kl_weight)

        # IWAE loss: -log E[p(x|z) p(z) / q(z|x)]
        # Here we ignore the constant shift of -log(k_samples)
        loss = -elbo.logsumexp(0).mean()

        x_recon = x_recon.view(-1, *x.shape)
        loss_breakdown = {
            'loss': loss.item(),
            'KL': kl_loss.mean().item(),
            'recon': recon_loss.mean().item(),
        }
        return loss, z_T, x_recon, elbo, loss_breakdown

    def impute(self, x, mask, k_samples=10):
        self.eval()
        with torch.no_grad():
            _, z, x_recon, elbo, _ = self(x, mask, k_samples)
            # sampling importance resampling
            is_idx = Categorical(logits=elbo.t()).sample()
            batch_idx = torch.arange(len(x))
            z = z[is_idx, batch_idx]
            x_recon = x_recon[is_idx, batch_idx]
        self.train()
        return x_recon


def train_pvae(args):
    torch.manual_seed(args.seed)

    if args.mask == 'indep':
        data = IndepMaskedCelebA(obs_prob=args.obs_prob)
        mask_str = f'{args.mask}_{args.obs_prob}'
    elif args.mask == 'block':
        data = BlockMaskedCelebA(block_len=args.block_len)
        mask_str = f'{args.mask}_{args.block_len}'

    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                             drop_last=True)

    test_loader = DataLoader(data, batch_size=args.batch_size, drop_last=True)

    decoder = ConvDecoder(args.latent)
    encoder = ConvEncoder(args.latent, args.flow, logprob=True)
    pvae = PVAE(encoder, decoder).to(device)

    optimizer = optim.Adam(pvae.parameters(), lr=args.lr)

    scheduler = None
    if args.min_lr is not None:
        lr_steps = 10
        step_size = args.epoch // lr_steps
        gamma = (args.min_lr / args.lr)**(1 / lr_steps)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

    rand_z = torch.empty(args.batch_size, args.latent, device=device)

    path = '{}_{}_{}'.format(
        args.prefix, datetime.now().strftime('%m%d.%H%M%S'), mask_str)
    output_dir = Path('results') / 'celeba-pvae' / path
    mkdir(output_dir)
    print(output_dir)

    if args.save_interval > 0:
        model_dir = mkdir(output_dir / 'model')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(output_dir / 'log.txt'),
            logging.StreamHandler(sys.stdout),
        ],
    )

    with (output_dir / 'args.txt').open('w') as f:
        print(pprint.pformat(vars(args)), file=f)

    vis = Visualizer(output_dir, loss_xlim=(0, args.epoch))

    test_x, test_mask, index = iter(test_loader).next()
    test_x = test_x.to(device)
    test_mask = test_mask.to(device).float()
    bbox = None
    if data.mask_loc is not None:
        bbox = [data.mask_loc[idx] for idx in index]

    kl_center = (args.kl_on + args.kl_off) / 2
    kl_scale = 1 / min(args.kl_on - args.kl_off, 1)

    for epoch in range(args.epoch):
        if epoch >= args.kl_on:
            kl_weight = 1
        elif epoch < args.kl_off:
            kl_weight = 0
        else:
            kl_weight = 1 / (1 + math.exp(-(epoch - kl_center) * kl_scale))
        loss_breakdown = defaultdict(float)
        for x, mask, _ in data_loader:
            x = x.to(device)
            mask = mask.to(device).float()

            optimizer.zero_grad()
            loss, _, _, _, loss_info = pvae(
                x, mask, args.k, kl_weight, args.ae)
            loss.backward()
            optimizer.step()
            for name, val in loss_info.items():
                loss_breakdown[name] += val

        if scheduler:
            scheduler.step()

        vis.plot_loss(epoch, loss_breakdown)

        if epoch % args.plot_interval == 0:
            x_recon = pvae.impute(test_x, test_mask, args.k)
            with torch.no_grad():
                pvae.eval()
                rand_z.normal_()
                _, x_gen = decoder(rand_z)
                pvae.train()
            vis.plot(epoch, test_x, test_mask, bbox, x_recon, x_gen)

        model_dict = {
            'pvae': pvae.state_dict(),
            'history': vis.history,
            'epoch': epoch,
            'args': args,
        }
        torch.save(model_dict, str(output_dir / 'model.pth'))
        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            torch.save(model_dict, str(model_dir / f'{epoch:04d}.pth'))

    print(output_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=3)
    # training options
    parser.add_argument('--plot-interval', type=int, default=50)
    parser.add_argument('--save-interval', type=int, default=50)
    # mask options (data): block|indep
    parser.add_argument('--mask', default='block')
    # option for block: set to 0 for variable size
    parser.add_argument('--block-len', type=int, default=32)
    # option for indep:
    parser.add_argument('--obs-prob', type=float, default=.2)

    parser.add_argument('--flow', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=None)

    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--prefix', default='pvae')
    parser.add_argument('--latent', type=int, default=128)
    parser.add_argument('--kl-off', type=int, default=10)
    # set --kl-on to 0 to use constant kl_weight = 1
    parser.add_argument('--kl-on', type=int, default=20)
    parser.add_argument('--ae', type=float, default=1)

    args = parser.parse_args()

    train_pvae(args)


if __name__ == '__main__':
    main()
