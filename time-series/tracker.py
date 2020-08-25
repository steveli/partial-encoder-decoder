import torch
from collections import defaultdict
import logging
import sys


class Tracker:
    def __init__(self, log_dir, n_train_batch):
        self.log_dir = log_dir
        self.n_train_batch = n_train_batch
        self.loss = defaultdict(list)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_dir / 'log.txt'),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.print_header = True

    def log(self, epoch, loss_breakdown, epoch_time, time_elapsed):
        for loss_name, loss_val in loss_breakdown.items():
            self.loss[loss_name].append(loss_val / self.n_train_batch)

        if self.print_header:
            logging.info(' ' * 7 + '  '.join(
                f'{key:>12}' for key in sorted(self.loss)))
            self.print_header = False
        logging.info(f'[{epoch:4}] ' + '  '.join(
            f'{val[-1]:12.4f}' for _, val in sorted(self.loss.items())))

        torch.save(self.loss, str(self.log_dir / 'log.pth'))

        with (self.log_dir / 'time.txt').open('a') as f:
            print(epoch, epoch_time, time_elapsed, file=f)
