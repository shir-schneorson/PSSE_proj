
import torch
import math

def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T+1, dtype=torch.float64)
    f = torch.cos(((steps/T + s) / (1+s)) * math.pi / 2) ** 2
    alphas_cumprod = (f / f[0]).clamp(min=1e-10, max=1.0)
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.float().clamp(1e-5, 0.999)

def make_schedule(T, kind="cosine"):
    if kind == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        raise NotImplementedError(kind)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars
