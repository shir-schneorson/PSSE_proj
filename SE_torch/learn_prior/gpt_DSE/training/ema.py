
import copy
import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        ssd = self.shadow.state_dict()
        for k in ssd.keys():
            ssd[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)

    def unwrap(self):
        return self.shadow
