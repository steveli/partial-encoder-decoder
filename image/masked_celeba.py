import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


class MaskedCelebA(datasets.ImageFolder):
    def __init__(self, data_dir='celeba-data', image_size=64, random_seed=0):
        transform = transforms.Compose([
            transforms.CenterCrop(108),
            transforms.Resize(size=image_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
        ])

        super().__init__(data_dir, transform)

        self.rnd = np.random.RandomState(random_seed)
        torch.manual_seed(random_seed)
        self.image_size = image_size
        self.cache_images()
        self.generate_masks()
        self.mask = torch.stack(self.mask)

    def cache_images(self):
        images = []
        for i in range(len(self)):
            image, _ = super().__getitem__(i)
            images.append(image)
        self.images = torch.stack(images)

    def __getitem__(self, index):
        return self.images[index], self.mask[index], index

    def __len__(self):
        return super().__len__()


class BlockMaskedCelebA(MaskedCelebA):
    def __init__(self, block_len=None, *args, **kwargs):
        self.block_len = block_len
        super().__init__(*args, **kwargs)

    def generate_masks(self):
        d0_len = d1_len = self.image_size
        d0_min_len = 12
        d0_max_len = d0_len - d0_min_len
        d1_min_len = 12
        d1_max_len = d1_len - d1_min_len

        n_masks = len(self)
        self.mask = [None] * n_masks
        self.mask_loc = [None] * n_masks
        for i in range(n_masks):
            if self.block_len == 0:
                d0_mask_len = self.rnd.randint(d0_min_len, d0_max_len)
                d1_mask_len = self.rnd.randint(d1_min_len, d1_max_len)
            else:
                d0_mask_len = d1_mask_len = self.block_len

            d0_start = self.rnd.randint(0, d0_len - d0_mask_len + 1)
            d1_start = self.rnd.randint(0, d1_len - d1_mask_len + 1)

            mask = torch.zeros((d0_len, d1_len), dtype=torch.uint8)
            mask[d0_start:(d0_start + d0_mask_len),
                 d1_start:(d1_start + d1_mask_len)] = 1
            self.mask[i] = mask[None]
            self.mask_loc[i] = d0_start, d1_start, d0_mask_len, d1_mask_len


class IndepMaskedCelebA(MaskedCelebA):
    def __init__(self, obs_prob=.2, obs_prob_max=None, *args, **kwargs):
        self.prob = obs_prob
        self.prob_max = obs_prob_max
        self.mask_loc = None
        super().__init__(*args, **kwargs)

    def generate_masks(self):
        imsize = self.image_size
        prob = self.prob
        prob_max = self.prob_max
        n_masks = len(self)
        self.mask = [None] * n_masks
        for i in range(n_masks):
            if prob_max is None:
                p = prob
            else:
                p = self.rnd.uniform(prob, prob_max)
            self.mask[i] = torch.ByteTensor(1, imsize, imsize).bernoulli_(p)
