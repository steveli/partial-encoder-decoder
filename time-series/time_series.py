import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
from sklearn.model_selection import train_test_split


def kernel_width(max_time, cconv_ref, overlap_rate):
    return max_time / (cconv_ref + overlap_rate - overlap_rate * cconv_ref)


class TimeSeries(Dataset):
    def __init__(self, data, time, mask, label=None,
                 max_time=5, cconv_ref=98, overlap_rate=.5, device=None):
        self.data = torch.tensor(data, dtype=torch.float)
        self.time = torch.tensor(time, dtype=torch.float)
        self.mask = torch.tensor(mask, dtype=torch.float)

        if label is None:
            TimeSeries.__getitem__ = lambda self, index: (
                self.data[index], self.time[index], self.mask[index], index)
        else:
            self.label = torch.tensor(label, dtype=torch.float)
            TimeSeries.__getitem__ = lambda self, index: (
                self.data[index], self.time[index], self.mask[index],
                self.label[index], index)

        self.data_len, self.channels = self.data.shape[:2]
        self.cconv_ref = cconv_ref
        self.device = device
        k_width = kernel_width(max_time, cconv_ref, overlap_rate)
        margin = k_width / 2
        refs = torch.linspace(margin, max_time - margin, cconv_ref)

        self.pseudo, self.deg, self.ref_idx, self.t_idx = [
            [[None] * self.channels for _ in range(self.data_len)]
            for _ in range(4)]

        for i, (y, t, m) in enumerate(zip(self.data, self.time, self.mask)):
            for c in range(self.channels):
                tc = t[c][m[c] == 1]
                dis = (tc - refs[:, None]) / k_width + .5
                dmask = (dis <= 1) * (dis >= 0)
                self.ref_idx[i][c], self.t_idx[i][c] = torch.nonzero(dmask).t()
                # Pseudo coordinates in [0, 1]
                self.pseudo[i][c] = dis[dmask]
                cur_deg = torch.zeros(self.cconv_ref)
                cur_deg.scatter_add_(0, self.ref_idx[i][c],
                                     torch.ones(self.ref_idx[i][c].shape))
                self.deg[i][c] = cur_deg.clamp(min=1)

    def __len__(self):
        return self.data_len

    def make_graph(self, data, time, mask, index):
        pseudo = [
            torch.cat([self.pseudo[idx][c] for idx in index])
            .to(self.device).unsqueeze_(1).requires_grad_(False)
            for c in range(self.channels)]

        # Indices accumulated across mini-batch. Used for adding
        # convolution results to linearized padded tensor.
        cum_ref_idx = [
            torch.cat([self.ref_idx[idx][c] + i * self.cconv_ref
                       for i, idx in enumerate(index)])
            .to(self.device).unsqueeze_(1).requires_grad_(False)
            for c in range(self.channels)]

        concat_y = [
            torch.cat(
                [y[c][(m[c] == 1).requires_grad_(False)][self.t_idx[idx][c]]
                 for y, m, idx in zip(data, mask, index)])
            .to(self.device).unsqueeze_(1)
            for c in range(self.channels)]

        deg = [
            torch.cat([self.deg[idx][c] for idx in index])
            .to(self.device).unsqueeze_(1).requires_grad_(False)
            for c in range(self.channels)]

        return pseudo, cum_ref_idx, concat_y, deg

    def collate_fn(self, batch):
        batch = [x.to(self.device) for x in default_collate(batch)]
        # For labeled data, skip the label as the 4th entry.
        (data, time, mask), index = batch[:3], batch[-1]
        graph = self.make_graph(data, time, mask, index)
        return batch + [graph]


def split_data(data_file, rnd, max_time, cconv_ref, overlap, device,
               rescale=False):
    raw_data = np.load(data_file)

    if len(raw_data) == 4:
        time_np = raw_data['time']
        data_np = raw_data['data']
        mask_np = raw_data['mask']
        label_np = raw_data['label'].squeeze()

        (tv_time, test_time, tv_data, test_data,
         tv_mask, test_mask, tv_label, test_label) = train_test_split(
             time_np, data_np, mask_np, label_np,
             train_size=.8, stratify=label_np, random_state=rnd)

        (train_time, val_time, train_data, val_data,
         train_mask, val_mask, train_label, val_label) = train_test_split(
             tv_time, tv_data, tv_mask, tv_label,
             train_size=.8, stratify=tv_label, random_state=rnd)

    elif len(raw_data) == 8:
        tv_time = raw_data['train_time']
        tv_data = raw_data['train_data']
        tv_mask = raw_data['train_mask']
        tv_label = raw_data['train_label']

        test_time = raw_data['test_time']
        test_data = raw_data['test_data']
        test_mask = raw_data['test_mask']
        test_label = raw_data['test_label']

        (train_time, val_time, train_data, val_data,
         train_mask, val_mask, train_label, val_label) = train_test_split(
             tv_time, tv_data, tv_mask, tv_label,
             train_size=.8, stratify=tv_label, random_state=rnd)

    elif len(raw_data) == 12:
        train_time = raw_data['train_time']
        train_data = raw_data['train_data']
        train_mask = raw_data['train_mask']
        train_label = raw_data['train_label']

        test_time = raw_data['test_time']
        test_data = raw_data['test_data']
        test_mask = raw_data['test_mask']
        test_label = raw_data['test_label']

        val_time = raw_data['val_time']
        val_data = raw_data['val_data']
        val_mask = raw_data['val_mask']
        val_label = raw_data['val_label']
    else:
        raise Exception('Invalid data')

    # Scale time
    train_time *= max_time
    test_time *= max_time
    val_time *= max_time

    # Rescale data from [0, 1] to [-1, 1]
    if rescale:
        train_data = 2 * train_data - 1
        val_data = 2 * val_data - 1
        test_data = 2 * test_data - 1

    train_dataset = TimeSeries(
        train_data, train_time, train_mask, train_label, max_time=max_time,
        cconv_ref=cconv_ref, overlap_rate=overlap, device=device)

    val_dataset = TimeSeries(
        val_data, val_time, val_mask, val_label, max_time=max_time,
        cconv_ref=cconv_ref, overlap_rate=overlap, device=device)

    test_dataset = TimeSeries(
        test_data, test_time, test_mask, test_label, max_time=max_time,
        cconv_ref=cconv_ref, overlap_rate=overlap, device=device)

    return train_dataset, val_dataset, test_dataset
