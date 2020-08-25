import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np


class MaskedMNIST(Dataset):
    def __init__(self, data_dir='mnist-data', image_size=28, random_seed=0):
        self.rnd = np.random.RandomState(random_seed)
        torch.manual_seed(random_seed)
        self.image_size = image_size
        if image_size == 28:
            self.data = datasets.MNIST(
                data_dir, train=True, download=True,
                transform=transforms.ToTensor())
        else:
            self.data = datasets.MNIST(
                data_dir, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size), transforms.ToTensor()]))
        self.generate_masks()

    def __getitem__(self, index):
        image, label = self.data[index]
        mask = self.mask[index]
        return image * mask.float(), mask[None], index

    def __len__(self):
        return len(self.data)

    def generate_masks(self):
        raise NotImplementedError


class BlockMaskedMNIST(MaskedMNIST):
    def __init__(self, block_len=11, block_len_max=None, *args, **kwargs):
        self.block_len = block_len
        self.block_len_max = block_len_max
        super().__init__(*args, **kwargs)

    def generate_masks(self):
        d0_len = d1_len = self.image_size
        n_masks = len(self)
        self.mask = [None] * n_masks
        self.mask_loc = [None] * n_masks
        for i in range(n_masks):
            if self.block_len_max is None:
                d0_mask_len = d1_mask_len = self.block_len
            else:
                d0_mask_len = self.rnd.randint(
                    self.block_len, self.block_len_max)
                d1_mask_len = self.rnd.randint(
                    self.block_len, self.block_len_max)

            d0_start = self.rnd.randint(0, d0_len - d0_mask_len + 1)
            d1_start = self.rnd.randint(0, d1_len - d1_mask_len + 1)

            mask = torch.zeros((d0_len, d1_len), dtype=torch.uint8)
            mask[d0_start:(d0_start + d0_mask_len),
                 d1_start:(d1_start + d1_mask_len)] = 1
            self.mask[i] = mask
            self.mask_loc[i] = d0_start, d1_start, d0_mask_len, d1_mask_len


class IndepMaskedMNIST(MaskedMNIST):
    def __init__(self, obs_prob=.2, obs_prob_max=None, *args, **kwargs):
        self.prob = obs_prob
        self.prob_max = obs_prob_max
        self.mask_loc = None
        super().__init__(*args, **kwargs)

    def generate_masks(self):
        imsize = self.image_size
        n_masks = len(self)
        self.mask = [None] * n_masks
        for i in range(n_masks):
            if self.prob_max is None:
                p = self.prob
            else:
                p = self.rnd.uniform(self.prob, self.prob_max)
            self.mask[i] = torch.ByteTensor(imsize, imsize).bernoulli_(p)
