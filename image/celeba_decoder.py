import torch
import torch.nn as nn


def dconv_bn_relu(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                           padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU())


class ConvDecoder(nn.Module):
    def __init__(self, latent_size=128):
        super().__init__()

        self.out_channels = 3
        dim = 64

        self.l1 = nn.Sequential(
            nn.Linear(latent_size, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())

        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, self.out_channels, 5, 2,
                               padding=2, output_padding=1))

    def forward(self, input):
        x = self.l1(input)
        x = x.view(x.shape[0], -1, 4, 4)
        x = self.l2_5(x)
        return x, torch.sigmoid(x)
