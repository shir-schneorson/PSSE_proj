import torch


def linear_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


class VarScheduler:
    def __init__(self, num_steps, device='cpu', dtype=torch.float32, schedule_type='linear'):
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        self.device = device
        self.dtype = dtype
        self.betas = self._create_betas().to(device=self.device, dtype=self.dtype)
        self.alphas = self._create_alphas().to(device=self.device, dtype=self.dtype)
        self.alpha_bars = self._create_alpha_bars().to(device=self.device, dtype=self.dtype)


    def _create_betas(self):
        if self.schedule_type == 'linear':
            return linear_beta_schedule(self.num_steps)
        else:
            raise NotImplementedError(self.schedule_type)

    def _create_alphas(self):
        return 1 - self.betas

    def _create_alpha_bars(self):
        return torch.cumprod(self.alphas, dim=0)

    def __getitem__(self, item):
        return self.betas[item], self.alphas[item], self.alpha_bars[item]