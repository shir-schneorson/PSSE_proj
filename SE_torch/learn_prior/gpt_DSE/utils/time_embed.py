
import torch
import torch.nn as nn
import math

class FourierTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t_idx, T):
        """Discrete timestep embedding -> [*, dim]
        t_idx: [N] integers in [0..T-1]
        """
        device = t_idx.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1 + 1e-8))
        )
        angles = t_idx.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb
