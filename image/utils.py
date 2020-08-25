def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path
