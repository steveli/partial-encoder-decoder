import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import logging
import sys
from collections import defaultdict
from pathlib import Path
from utils import mkdir


class Visualizer(object):
    def __init__(self,
                 output_dir,
                 loss_xlim=None,
                 loss_ylim=None):
        self.output_dir = Path(output_dir)
        self.recons_dir = mkdir(self.output_dir / 'recons')
        self.gen_dir = mkdir(self.output_dir / 'gen')
        self.loss_xlim = loss_xlim
        self.loss_ylim = loss_ylim
        sns.set()
        self.rows, self.cols = 8, 16
        self.at_start = True

        self.history = defaultdict(list)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(output_dir / 'log.txt'),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.print_header = True

    def plot_subgrid(self, images, bbox=None, filename=None):
        rows, cols = 8, 16
        scale = .75
        fig, ax = plt.subplots(figsize=(cols * scale, rows * scale))
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])

        inner_grid = gridspec.GridSpec(rows, cols, fig, wspace=.05, hspace=.05,
                                       left=0, right=1, top=1, bottom=0)

        images = images[:(rows * cols)].cpu().numpy()
        if images.shape[1] == 1:   # single channel
            images = images.squeeze(1)
            cmap = 'binary_r'
        else:   # 3 channels
            images = images.transpose((0, 2, 3, 1))
            cmap = None

        for i, image in enumerate(images):
            ax = plt.Subplot(fig, inner_grid[i])
            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            ax.imshow(image, interpolation='none', aspect='equal',
                      cmap=cmap, vmin=0, vmax=1)

            if bbox is not None:
                d0, d1, d0_len, d1_len = bbox[i]
                ax.add_patch(Rectangle(
                    (d1 - .5, d0 - .5), d1_len, d0_len, lw=1,
                    edgecolor='red', fill=False))
            fig.add_subplot(ax)

        if filename is not None:
            plt.savefig(str(filename))
        plt.close(fig)

    def plot(self, epoch, x, mask, bbox, x_recon, x_gen):
        if self.at_start:
            self.plot_subgrid(x * mask + .5 * (1 - mask), bbox,
                              self.recons_dir / f'groundtruth.png')
            self.at_start = False
        self.plot_subgrid(x * mask + x_recon * (1 - mask), bbox,
                          self.recons_dir / f'{epoch:04d}.png')
        self.plot_subgrid(x_gen, None, self.gen_dir / f'{epoch:04d}.png')

    def plot_loss(self, epoch, losses):
        for name, val in losses.items():
            self.history[name].append(val)

        fig, ax_trace = plt.subplots(figsize=(6, 4))
        ax_trace.set_ylabel('loss')
        ax_trace.set_xlabel('epochs')
        if self.loss_xlim is not None:
            ax_trace.set_xlim(self.loss_xlim)
        if self.loss_ylim is not None:
            ax_trace.set_ylim(self.loss_ylim)
        for label, loss in self.history.items():
            ax_trace.plot(loss, '-', label=label)
        if len(self.history) > 1:
            ax_trace.legend(ncol=len(self.history), loc='upper center')
        plt.tight_layout()
        plt.savefig(str(self.output_dir / 'loss.png'), dpi=300)
        plt.close(fig)

        if self.print_header:
            logging.info(' ' * 7 + '  '.join(
                f'{key:>12}' for key in sorted(losses)))
            self.print_header = False
        logging.info(f'[{epoch:4}] ' + '  '.join(
            f'{val:12.4f}' for _, val in sorted(losses.items())))
