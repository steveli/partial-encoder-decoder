import torch


def mmd(x, y):
    n, dim = x.shape

    xx = (x**2).sum(1, keepdim=True)
    yy = (y**2).sum(1, keepdim=True)

    outer_xx = torch.mm(x, x.t())
    outer_yy = torch.mm(y, y.t())
    outer_xy = torch.mm(x, y.t())

    diff_xx = xx + xx.t() - 2 * outer_xx
    diff_yy = yy + yy.t() - 2 * outer_yy
    diff_xy = xx + yy.t() - 2 * outer_xy

    C = 2. * dim
    k_xx = C / (C + diff_xx)
    k_yy = C / (C + diff_yy)
    k_xy = C / (C + diff_xy)

    mean_xx = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1))
    mean_yy = (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1))
    mean_xy = k_xy.sum() / (n * n)

    return mean_xx + mean_yy - 2 * mean_xy
