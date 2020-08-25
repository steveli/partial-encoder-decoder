import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from utils import mkdir, to_numpy


sns.set()

sns.set_style('darkgrid', {
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.right': True,
    'axes.spines.top': True,
    'patch.edgecolor': 'k',
    'axes.edgecolor': '.1',
    'axes.facecolor': '.95',
})


def plot_samples(data_unif, time_unif, data=None, time=None, mask=None,
                 rescaler=None, max_time=1, img_path=None, nrows=2, ncols=4):
    data_unif = to_numpy(data_unif)
    time_unif = to_numpy(time_unif)
    n_channels = data_unif.shape[1]
    if rescaler:
        data_unif = rescaler.unrescale(data_unif)

    if data is not None:
        data = to_numpy(data)
        time = to_numpy(time)
        mask = to_numpy(mask)
        if rescaler:
            data = rescaler.unrescale(data)

    fig = plt.figure(figsize=(6 * ncols, 2 * n_channels * nrows))
    gs = gridspec.GridSpec(nrows, ncols, wspace=.1, hspace=.1)

    for i in range(nrows * ncols):
        outer_ax = plt.subplot(gs[i])
        outer_ax.set_xticks([])
        outer_ax.set_yticks([])

        inner_grid = gridspec.GridSpecFromSubplotSpec(
            n_channels, 1, subplot_spec=gs[i], hspace=0)
        for k in range(n_channels):
            ax = plt.Subplot(fig, inner_grid[k])
            ax.plot(time_unif[i, k], data_unif[i, k],
                    'k-', alpha=.5, linewidth=.6)
            if data is not None:
                ax.scatter(time[i, k, mask[i, k] == 1],
                           data[i, k, mask[i, k] == 1], c='r',
                           s=30,
                           linewidth=1.5,
                           marker='x')
            ax.set_ylim(-1.2, 1.2)
            ax.set_xlim(0, max_time)
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            fig.add_subplot(ax)

    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)


class Visualizer:
    def __init__(self, encoder, decoder, batch_size, max_time, test_loader,
                 rescaler, output_dir, device):
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.rescaler = rescaler
        (self.test_val, self.test_idx, self.test_mask,
         _, self.test_cconv_graph) = next(iter(test_loader))
        in_channels = self.test_val.shape[1]
        self.max_time = max_time
        t = torch.linspace(0, max_time, 200, device=device)
        self.t = t.expand(batch_size, in_channels, len(t)).contiguous()
        self.t_mask = torch.ones_like(self.t)

        self.gen_data_dir = mkdir(output_dir / 'gen')
        self.imp_data_dir = mkdir(output_dir / 'imp')

    def plot(self, epoch):
        filename = f'{epoch:04d}.png'

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            z = self.encoder(self.test_cconv_graph, self.batch_size)
            if not torch.is_tensor(z):   # P-VAE encoder returns a list
                z = z[0]
            imp_data = self.decoder(z, self.t, self.t_mask)
            plot_samples(imp_data, self.t,
                         self.test_val, self.test_idx, self.test_mask,
                         rescaler=self.rescaler,
                         max_time=self.max_time,
                         img_path=f'{self.imp_data_dir / filename}')

            data_noise = torch.empty_like(z).normal_()
            gen_data = self.decoder(data_noise, self.t, self.t_mask)
            plot_samples(gen_data, self.t,
                         rescaler=self.rescaler,
                         max_time=self.max_time,
                         img_path=f'{self.gen_data_dir / filename}')
        self.decoder.train()
        self.encoder.train()
