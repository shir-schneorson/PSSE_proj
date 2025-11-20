import os

import torch

class ChannelScaler:
    """Per-channel standardization for node_feats: [V, sinθ, cosθ]."""
    def __init__(self, path=None, mean=None, std=None, eps=1e-10):
        if path is not None and os.path.exists(path):
            self.mean = torch.load(f'{path}/mean.pt')
            self.std = torch.load(f'{path}/std.pt')

        else:
            self.mean = mean
            self.std = std

        self.eps = eps

    def fit(self, x):
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)
        self.std[self.std <= self.eps] = 1

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.mean, f'{path}/mean.pt')
        torch.save(self.std, f'{path}/std.pt')

