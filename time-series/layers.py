import torch
import torch.nn as nn
import torch.nn.functional as F


class ResLinearBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.Dropout(),
            nn.LeakyReLU(.2),
            nn.Linear(out_size, out_size),
            nn.Dropout(),
            nn.LeakyReLU(.2),
        )

        self.skip = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(.2),
        )

    def forward(self, x):
        return self.linear(x) + self.skip(x)


class Classifier(nn.Module):
    def __init__(self, in_size, layers=1):
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks.append(ResLinearBlock(in_size, in_size))
        # No spectral normalization for the last layer
        blocks.append(nn.Linear(in_size, 1))
        self.res_linear = nn.Sequential(*blocks)

    def forward(self, x):
        return self.res_linear(x)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.activation = nn.ReLU(inplace=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1)
        self.convx = None
        if in_channels != out_channels:
            self.convx = nn.Conv1d(
                in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.activation(self.bn1(x))
        h = F.interpolate(h, scale_factor=2)
        x = F.interpolate(x, scale_factor=2)
        if self.convx:
            x = self.convx(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        return h + x


class GridDecoder(nn.Module):
    def __init__(self, dim_z, channels, start_len=16, squash=None):
        super().__init__()
        self.activation = nn.ReLU(inplace=False)
        self.start_len = start_len
        self.linear = nn.Linear(dim_z, channels[0] * start_len)
        self.blocks = nn.Sequential(
            *[GBlock(in_channels, channels[c + 1])
              for c, in_channels in enumerate(channels[:-2])])
        self.output = nn.Sequential(
            nn.BatchNorm1d(channels[-2]),
            self.activation,
            nn.Conv1d(channels[-2], channels[-1], kernel_size=3, padding=1),
        )
        self.squash = squash

    def forward(self, z):
        h = self.linear(z)
        h = h.view(h.shape[0], -1, self.start_len)
        h = self.blocks(h)
        h = self.output(h)
        if self.squash:
            h = self.squash(h)
        return h


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.activation = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1)
        self.convx = None
        if in_channels != out_channels:
            self.convx = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.downsample = None
        if downsample:
            self.downsample = nn.AvgPool1d(2)

    def shortcut(self, x):
        if self.convx:
            x = self.convx(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def forward(self, x):
        # pre-activation
        h = self.activation(x)
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)
        return h + self.shortcut(x)


class GridEncoder(nn.Module):
    def __init__(self, channels, out_dim=1):
        super().__init__()
        self.activation = nn.ReLU(inplace=False)
        self.blocks = nn.Sequential(
            *[DBlock(in_channels, out_channels)
              for in_channels, out_channels
              in zip(channels[:-1], channels[1:])])
        self.linear = nn.Linear(channels[-1], out_dim)

    def forward(self, x):
        h = x
        h = self.blocks(h)
        h = self.activation(h).sum(2)
        return self.linear(h)


class Decoder(nn.Module):
    def __init__(self, grid_decoder, max_time=5, kernel_bw=None, dec_ref=128):
        super().__init__()
        if kernel_bw is None:
            self.kernel_bw = max_time / dec_ref * 3
        else:
            self.kernel_bw = kernel_bw
        # ref_times are the assigned time stamps for the evenly-spaced
        # generated sequences by conv1d.
        self.register_buffer('ref_times', torch.linspace(0, max_time, dec_ref))
        self.ref_times = self.ref_times[:, None]
        self.grid_decoder = grid_decoder

    def forward(self, code, time, mask):
        """
        Args:
            code: shape (batch_size, latent_size)
            time: shape (batch_size, channels, max_seq_len)
            mask: shape (batch_size, channels, max_seq_len)

        Returns:
            interpolated tensor of shape (batch_size, max_seq_len)
        """
        # shape of x: (batch_size, n_channels, dec_ref)
        x = self.grid_decoder(code)

        # t_diff shape: (batch_size, n_channels, dec_ref, max_seq_len)
        t_diff = time[:, :, None] - self.ref_times

        # Epanechnikov quadratic kernel:
        # K_\lambda(x_0, x) = relu(3/4 * (1 - (|x_0 - x| / \lambda)^2))
        # shape of w: (batch_size, n_channels, dec_ref, max_seq_len)
        w = F.relu((1 - (t_diff / self.kernel_bw)**2) * .75)
        # avoid divided by zero
        # normalizer = torch.clamp(w.sum(2), min=1e-6)
        # return ((x[:, :, :, None] * w).sum(2) * mask) / normalizer
        ks_x = ((x[:, :, :, None] * w).sum(2) * mask) / w.sum(2)
        return ks_x


def gan_loss(real, fake, real_target, fake_target):
    real_score = sum(F.binary_cross_entropy_with_logits(
        r, r.new_tensor(real_target).expand_as(r)) for r in real)
    fake_score = sum(F.binary_cross_entropy_with_logits(
        f, f.new_tensor(fake_target).expand_as(f)) for f in fake)
    return real_score + fake_score
