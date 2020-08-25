class EMA:
    def __init__(self, model, decay, start_iter=0):
        self.model = model
        self.beta = 1 - decay
        self.start_iter = start_iter
        self.iter = 0
        self.shadow = None

    def state_dict(self):
        """Returns the EMA state as a dictionary for serialization."""
        # NOTE: skip saving `model`
        return {
            'beta': self.beta,
            'start_iter': self.start_iter,
            'iter': self.iter,
            'shadow': self.shadow,
        }

    def update(self):
        if self.iter < self.start_iter:
            self.iter += 1
        else:
            if self.shadow is None:
                self.shadow = {}
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.shadow[name] = param.data.clone()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # p = p - (1 - delay) * (p - theta)
                    #   = delay * p + (1 - delay) * theta
                    self.shadow[name].sub_(
                        self.beta * (self.shadow[name] - param.data))

    def apply(self):
        if self.shadow is None:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name], param.data = param.data, self.shadow[name]

    def restore(self):
        self.apply()
