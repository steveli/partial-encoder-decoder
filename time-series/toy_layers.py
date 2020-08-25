import torch.nn as nn
from torch.nn.utils import spectral_norm


def dconv_bn_relu(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose1d(in_dim, out_dim, 5, 2,
                           padding=2, output_padding=1, bias=False),
        nn.BatchNorm1d(out_dim),
        nn.ReLU())


class SeqGeneratorDiscrete(nn.Module):
    def __init__(self, n_channels=3, latent_size=128, squash=None):
        super().__init__()

        self.l1 = nn.Sequential(
            nn.Linear(latent_size, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU())

        self.l2 = nn.Sequential(
            dconv_bn_relu(256, 128),
            dconv_bn_relu(128, 64),
            dconv_bn_relu(64, 32),
            nn.ConvTranspose1d(32, n_channels, 5, 2,
                               padding=2, output_padding=1))
        self.squash = squash

    def forward(self, z):
        h = self.l1(z)
        h = h.view(h.shape[0], -1, 8)
        h = self.l2(h)
        if self.squash:
            h = self.squash(h)
        return h


def conv_ln_lrelu(in_dim, out_dim):
    return nn.Sequential(
        spectral_norm(nn.Conv1d(in_dim, out_dim, 5, 2, 2)),
        nn.LeakyReLU(0.2))
