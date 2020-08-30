from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import logging
from pathlib import Path
from datetime import datetime
import pprint
import argparse
from masked_mnist import IndepMaskedMNIST, BlockMaskedMNIST
from mnist_decoder import ConvDecoder
from mnist_encoder import ConvEncoder
from utils import mkdir, make_scheduler
from visualize import Visualizer


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class PBiGAN(nn.Module):
    def __init__(self, encoder, decoder, ae_loss='bce'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ae_loss = ae_loss

    def forward(self, x, mask, ae=True):
        z_T = self.encoder(x * mask)

        z_gen = torch.empty_like(z_T).normal_()
        x_gen_logit, x_gen = self.decoder(z_gen)

        x_logit, x_recon = self.decoder(z_T)

        recon_loss = None
        if ae:
            if self.ae_loss == 'mse':
                recon_loss = F.mse_loss(
                    x_recon * mask, x * mask, reduction='none') * mask
                recon_loss = recon_loss.sum((1, 2, 3)).mean()
            elif self.ae_loss == 'l1':
                recon_loss = F.l1_loss(
                    x_recon * mask, x * mask, reduction='none') * mask
                recon_loss = recon_loss.sum((1, 2, 3)).mean()
            elif self.ae_loss == 'smooth_l1':
                recon_loss = F.smooth_l1_loss(
                    x_recon * mask, x * mask, reduction='none') * mask
                recon_loss = recon_loss.sum((1, 2, 3)).mean()
            elif self.ae_loss == 'bce':
                # Bernoulli noise
                # recon_loss: -log p(x|z)
                recon_loss = F.binary_cross_entropy_with_logits(
                    x_logit * mask, x * mask, reduction='none') * mask
                recon_loss = recon_loss.sum((1, 2, 3)).mean()

        return z_T, z_gen, x_recon, x_gen, recon_loss


class ConvCritic(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.DIM = 64
        self.main = nn.Sequential(
            nn.Conv2d(1, self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(self.DIM, 2 * self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2 * self.DIM, 4 * self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
        )

        embed_size = 64

        self.z_fc = nn.Sequential(
            nn.Linear(latent_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_size, embed_size),
        )

        self.x_fc = nn.Linear(4 * 4 * 4 * self.DIM, embed_size)

        self.xz_fc = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.LayerNorm(embed_size),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_size, 1),
        )

    def forward(self, input):
        x, z = input
        x = x.view(-1, 1, 28, 28)
        x = self.main(x)
        x = x.view(x.shape[0], -1)
        x = self.x_fc(x)
        z = self.z_fc(z)
        xz = torch.cat((x, z), 1)
        xz = self.xz_fc(xz)
        return xz.view(-1)


class GradientPenalty:
    def __init__(self, critic, batch_size=64, gp_lambda=10):
        self.critic = critic
        self.gp_lambda = gp_lambda
        # Interpolation coefficient
        self.eps = torch.empty(batch_size, device=device)
        # For computing the gradient penalty
        self.ones = torch.ones(batch_size).to(device)

    def interpolate(self, real, fake):
        eps = self.eps.view([-1] + [1] * (len(real.shape) - 1))
        return (eps * real + (1 - eps) * fake).requires_grad_()

    def __call__(self, real, fake):
        real = [x.detach() for x in real]
        fake = [x.detach() for x in fake]
        self.eps.uniform_(0, 1)
        interp = [self.interpolate(a, b) for a, b in zip(real, fake)]
        grad_d = grad(self.critic(interp),
                      interp,
                      grad_outputs=self.ones,
                      create_graph=True)
        batch_size = real[0].shape[0]
        grad_d = torch.cat([g.view(batch_size, -1) for g in grad_d], 1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.gp_lambda
        return grad_penalty


def train_pbigan(args):
    torch.manual_seed(args.seed)

    if args.mask == 'indep':
        data = IndepMaskedMNIST(obs_prob=args.obs_prob,
                                obs_prob_max=args.obs_prob_max)
        mask_str = f'{args.mask}_{args.obs_prob}_{args.obs_prob_max}'
    elif args.mask == 'block':
        data = BlockMaskedMNIST(block_len=args.block_len,
                                block_len_max=args.block_len_max)
        mask_str = f'{args.mask}_{args.block_len}_{args.block_len_max}'

    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                             drop_last=True)
    mask_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                             drop_last=True)

    # Evaluate the training progress using 2000 examples from the training data
    test_loader = DataLoader(data, batch_size=args.batch_size, drop_last=True)

    decoder = ConvDecoder(args.latent)
    encoder = ConvEncoder(args.latent, args.flow, logprob=False)
    pbigan = PBiGAN(encoder, decoder, args.aeloss).to(device)

    critic = ConvCritic(args.latent).to(device)

    lrate = 1e-4
    optimizer = optim.Adam(pbigan.parameters(), lr=lrate, betas=(.5, .9))

    critic_optimizer = optim.Adam(
        critic.parameters(), lr=lrate, betas=(.5, .9))

    grad_penalty = GradientPenalty(critic, args.batch_size)

    scheduler = make_scheduler(optimizer, args.lr, args.min_lr, args.epoch)

    path = '{}_{}_{}'.format(
        args.prefix, datetime.now().strftime('%m%d.%H%M%S'), mask_str)
    output_dir = Path('results') / 'mnist-pbigan' / path
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

    vis = Visualizer(output_dir)

    test_x, test_mask, index = iter(test_loader).next()
    test_x = test_x.to(device)
    test_mask = test_mask.to(device).float()
    bbox = None
    if data.mask_loc is not None:
        bbox = [data.mask_loc[idx] for idx in index]

    n_critic = 5
    critic_updates = 0
    ae_weight = 0
    ae_flat = 100

    for epoch in range(args.epoch):
        loss_breakdown = defaultdict(float)

        if epoch > ae_flat:
            ae_weight = args.ae * (epoch - ae_flat) / (args.epoch - ae_flat)

        for (x, mask, _), (_, mask_gen, _) in zip(data_loader, mask_loader):
            x = x.to(device)
            mask = mask.to(device).float()
            mask_gen = mask_gen.to(device).float()

            z_enc, z_gen, x_rec, x_gen, _ = pbigan(x, mask, ae=False)

            real_score = critic((x * mask, z_enc)).mean()
            fake_score = critic((x_gen * mask_gen, z_gen)).mean()

            w_dist = real_score - fake_score
            D_loss = -w_dist + grad_penalty((x * mask, z_enc),
                                            (x_gen * mask_gen, z_gen))

            critic_optimizer.zero_grad()
            D_loss.backward()
            critic_optimizer.step()

            loss_breakdown['D'] += D_loss.item()

            critic_updates += 1

            if critic_updates == n_critic:
                critic_updates = 0

                # Update generators' parameters
                for p in critic.parameters():
                    p.requires_grad_(False)

                z_enc, z_gen, x_rec, x_gen, ae_loss = pbigan(x, mask)

                real_score = critic((x * mask, z_enc)).mean()
                fake_score = critic((x_gen * mask_gen, z_gen)).mean()

                G_loss = real_score - fake_score

                ae_loss = ae_loss * ae_weight
                loss = G_loss + ae_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_breakdown['G'] += G_loss.item()
                loss_breakdown['AE'] += ae_loss.item()
                loss_breakdown['total'] += loss.item()

                for p in critic.parameters():
                    p.requires_grad_(True)

        if scheduler:
            scheduler.step()

        vis.plot_loss(epoch, loss_breakdown)

        if epoch % args.plot_interval == 0:
            with torch.no_grad():
                pbigan.eval()
                z, z_gen, x_rec, x_gen, ae_loss = pbigan(test_x, test_mask)
                pbigan.train()
            vis.plot(epoch, test_x, test_mask, bbox, x_rec, x_gen)

        model_dict = {
            'pbigan': pbigan.state_dict(),
            'critic': critic.state_dict(),
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

    parser.add_argument('--seed', type=int, default=3,
                        help='random seed')
    # training options
    parser.add_argument('--plot-interval', type=int, default=50,
                        help='plot interval. 0 to disable plotting.')
    parser.add_argument('--save-interval', type=int, default=0,
                        help='interval to save models. 0 to disable saving.')
    parser.add_argument('--mask', default='block',
                        help='missing data mask. (options: block, indep)')
    # option for block: set to 0 for variable size
    parser.add_argument('--block-len', type=int, default=12,
                        help='size of observed block')
    parser.add_argument('--block-len-max', type=int, default=None,
                        help='max size of observed block. '
                             'Use fixed-size observed block if unspecified.')
    # option for indep:
    parser.add_argument('--obs-prob', type=float, default=.2,
                        help='observed probability for independent dropout')
    parser.add_argument('--obs-prob-max', type=float, default=None,
                        help='max observed probability for independent '
                             'dropout. Use fixed probability if unspecified.')

    parser.add_argument('--flow', type=int, default=2,
                        help='number of IAF layers')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--min-lr', type=float, default=-1,
                        help='min learning rate for LR scheduler. '
                             '-1 to disable annealing')

    parser.add_argument('--arch', default='conv',
                        help='network architecture. (options: fc, conv)')
    parser.add_argument('--epoch', type=int, default=2000,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--ae', type=float, default=.1,
                        help='autoencoding regularization strength')
    parser.add_argument('--prefix', default='pbigan',
                        help='prefix of output directory')
    parser.add_argument('--latent', type=int, default=128,
                        help='dimension of latent variable')
    parser.add_argument('--aeloss', default='bce',
                        help='autoencoding loss. '
                             '(options: mse, bce, smooth_l1, l1)')

    args = parser.parse_args()

    train_pbigan(args)


if __name__ == '__main__':
    main()
