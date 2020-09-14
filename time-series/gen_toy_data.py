import torch
import numpy as np
from torch.distributions.exponential import Exponential
import math


class HomogeneousPoissonProcess:
    def __init__(self, rate=1):
        self.rate = rate
        self.exp = Exponential(rate)

    def sample(self, size, max_seq_len, max_time=math.inf):
        gaps = self.exp.sample((size, max_seq_len))
        times = torch.cumsum(gaps, dim=1)
        masks = (times <= max_time).float()
        return times, masks


def gen_data(n_samples=10000, seq_len=200, max_time=1, poisson_rate=50,
             obs_span_rate=.25, save_file=None):
    """Generates a 3-channel synthetic dataset.

    The observations are within a window of size (max_time * obs_span_rate)
    randomly occurring at the time span [0, max_time].

    Args:
        n_samples:
            Number of data cases.
        seq_len:
            Maximum number of observations in a channel.
        max_time:
            Length of time interval [0, max_time].
        poisson_rate:
            Rate of homogeneous Poisson process.
        obs_span_rate:
            The continuous portion of the time span [0, max_time]
            that observations are restricted in.
        save_file:
            File name that the generated data is saved to.
    """
    n_channels = 3
    time_unif = np.linspace(0, max_time, seq_len)
    time_unif_3ch = np.broadcast_to(time_unif, (n_channels, seq_len))
    data_unif = np.empty((n_samples, n_channels, seq_len))
    sparse_data, sparse_time, sparse_mask = [
        np.empty((n_samples, n_channels, seq_len)) for _ in range(3)]
    tpp = HomogeneousPoissonProcess(rate=poisson_rate)

    def gen_time_series(offset1, offset2, t):
        t1 = t[0] + offset1
        t2 = t[2] + offset2
        t1_shift = t[1] + offset1 + 20
        data = np.empty((3, seq_len))
        data[0] = np.sin(t1 * 20 + np.sin(t1 * 20)) * .8
        data[1] = -np.sin(t1_shift * 20 + np.sin(t1_shift * 20)) * .5
        data[2] = np.sin(t2 * 12)
        return data

    for i in range(n_samples):
        offset1 = np.random.normal(0, 10)
        offset2 = np.random.uniform(0, 10)

        # Noise-free evenly-sampled time series
        data_unif[i] = gen_time_series(offset1, offset2, time_unif_3ch)

        # Generate observations between [0, obs_span_rate].
        times, masks = tpp.sample(3, seq_len, max_time=obs_span_rate)
        # Add independent random offset Unif(0, 1 - obs_span_rate) to each
        # channel so that all the observations will still be within [0, 1].
        times += torch.rand((3, 1)) * (1 - obs_span_rate)
        # Scale time span from [0, 1] to [0, max_time].
        times *= max_time
        # Set time entries corresponding to unobserved samples to time 0.
        sparse_time[i] = times * masks
        sparse_mask[i] = masks
        sparse_data[i] = gen_time_series(offset1, offset2, times)

    # Add a small independent Gaussian noise to each channel
    sparse_data += np.random.normal(0, .01, sparse_data.shape)

    # Pack the data to minimize the padded entries
    compact_len = sparse_mask.astype(int).sum(axis=2).max()
    compact_data, compact_time, compact_mask = [
        np.zeros((n_samples, 3, compact_len)) for _ in range(3)]
    for i in range(n_samples):
        for j in range(3):
            idx = sparse_mask[i, j] == 1
            n_obs = idx.sum()
            compact_data[i, j, :n_obs] = sparse_data[i, j, idx]
            compact_time[i, j, :n_obs] = sparse_time[i, j, idx]
            compact_mask[i, j, :n_obs] = sparse_mask[i, j, idx]

    if save_file:
        np.savez_compressed(
            save_file,
            time=compact_time,
            data=compact_data,
            mask=compact_mask,
            data_unif=data_unif,
            time_unif=time_unif,
        )

    return compact_data, compact_time, compact_mask, data_unif, time_unif


def main():
    gen_data(n_samples=10000, seq_len=200, max_time=1, poisson_rate=50,
             obs_span_rate=.25, save_file='toy-data.npz')


if __name__ == '__main__':
    main()
