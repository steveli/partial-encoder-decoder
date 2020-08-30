import torch.optim as optim


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
