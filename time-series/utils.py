import torch
import torch.optim as optim
import numpy as np


def to_numpy(v):
    if torch.is_tensor(v):
        return v.cpu().numpy()
    return v


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Rescaler:
    def __init__(self, data):
        channels = data.shape[1]
        ch_min = np.array([data[:, i].min() for i in range(channels)])
        ch_max = np.array([data[:, i].max() for i in range(channels)])
        self.ch_min, self.ch_max = ch_min[:, None], ch_max[:, None]

    def rescale(self, data):
        return 2 * (data - self.ch_min) / (self.ch_max - self.ch_min) - 1

    def unrescale(self, data):
        return .5 * (data + 1) * (self.ch_max - self.ch_min) + self.ch_min


def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_scheduler(optimizer, lr, min_lr, epochs, steps=10):
    if min_lr < 0:
        return None
    step_size = epochs // steps
    gamma = (min_lr / lr)**(1 / steps)
    return optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)
