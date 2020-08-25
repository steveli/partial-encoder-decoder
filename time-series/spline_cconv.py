import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torch_spline_conv import SplineBasis, SplineWeighting
import math


def kernel_width(max_time, ref_size, overlap_rate):
    return max_time / (ref_size + overlap_rate - overlap_rate * ref_size)


class ContinuousConv1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=64,
                 max_time=5,
                 ref_size=98,
                 overlap_rate=.5,
                 kernel_size=5,
                 norm=False,
                 bias=True,
                 spline_degree=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ref_size = ref_size
        self.overlap_rate = overlap_rate
        self.kernel_width = kernel_width(max_time, ref_size, overlap_rate)
        self.spline_degree = spline_degree
        self.norm = norm

        margin = self.kernel_width / 2
        refs = torch.linspace(margin, max_time - margin, ref_size)
        self.register_buffer('refs', refs)

        kernel_size = torch.tensor([kernel_size], dtype=torch.long)
        self.register_buffer('kernel_size', kernel_size)

        is_open_spline = torch.tensor([1], dtype=torch.uint8)
        self.register_buffer('is_open_spline', is_open_spline)

        self.weight = nn.Parameter(
            torch.Tensor(kernel_size, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_channels)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, pseudo, ref_idx, y, ref_deg, batch_size):
        conv_out = y[0].new_zeros(
            self.in_channels, self.ref_size * batch_size, self.out_channels)
        for c in range(self.in_channels):
            data = SplineBasis.apply(pseudo[c], self.kernel_size,
                                     self.is_open_spline, self.spline_degree)
            out = SplineWeighting.apply(
                y[c], self.weight[:, c].unsqueeze(1), *data)
            idx = ref_idx[c].expand_as(out)
            conv_out[c].scatter_add_(0, idx, out)
            if self.norm:
                conv_out[c].div_(ref_deg[c])
        conv_out = conv_out.sum(0)
        conv_out = conv_out.view(batch_size, self.ref_size, self.out_channels)
        conv_out = conv_out.transpose(1, 2)
        if self.bias is not None:
            conv_out = conv_out + self.bias[:, None]
        return conv_out


def gen_collate_fn(channels, max_time=5, ref_size=98, overlap_rate=.5,
                   device=None):
    k_width = kernel_width(max_time, ref_size, overlap_rate)
    margin = k_width / 2
    refs_ = torch.linspace(margin, max_time - margin, ref_size)

    def collate_fn(batch):
        y0 = batch[0][0]
        refs = refs_.to(y0.device)

        pseudo = [[] for _ in range(channels)]
        cum_ref_idx = [[] for _ in range(channels)]
        concat_y = [[] for _ in range(channels)]
        deg = [[] for _ in range(channels)]

        for i, ts_info in enumerate(batch):
            y, t, m = ts_info[:3]
            for c in range(channels):
                tc = t[c][m[c] == 1]
                yc = y[c][m[c] == 1]
                dis = (tc - refs[:, None]) / k_width + .5
                mask = (dis <= 1) * (dis >= 0)
                ref_idx, t_idx = torch.nonzero(mask).t()
                # Pseudo coordinates in [0, 1]
                pseudo[c].append(dis[mask])
                # Indices accumulated across mini-batch. Used for adding
                # convolution results to linearized padded tensor.
                cum_ref_idx[c].append(ref_idx + i * ref_size)
                concat_y[c].append(yc[t_idx])
                deg[c].append(y0.new_zeros(ref_size).scatter_add_(
                    0, ref_idx, y0.new_ones(ref_idx.shape)))

        for c in range(channels):
            pseudo[c] = torch.cat(pseudo[c]).unsqueeze(1).to(device)
            cum_ref_idx[c] = torch.cat(cum_ref_idx[c]).unsqueeze(1).to(device)
            concat_y[c] = torch.cat(concat_y[c]).unsqueeze(1).to(device)
            # clamp(min=1) to avoid dividing by zero
            deg[c] = torch.cat(deg[c]).clamp(min=1).unsqueeze(1).to(device)

        converted_batch = [x.to(device) for x in default_collate(batch)]
        return converted_batch + [(pseudo, cum_ref_idx, concat_y, deg)]

    return collate_fn
