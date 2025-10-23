import torch
import torch.nn as nn
from torch.optim import Adagrad, SGD, Muon, AdamW
from tqdm import tqdm


class WeightedSELoss(nn.Module):
    def __init__(self, z, R):
        super().__init__()
        self.z = z
        self.R = R

    def forward(self, z_est):
        error = (self.z - z_est).reshape(-1, 1)
        weighted_squared_error = error.T @ self.R @ error
        return weighted_squared_error


def SGD_se(x0, z, v, slk_bus, h_ac, nb, tol=1e-10, max_iter=1.5e+6, verbose=True):
    T = x0[:nb].clone()
    T[slk_bus[0]] = slk_bus[1]
    V = x0[nb:].clone()
    T.requires_grad_(True)
    V.requires_grad_(True)
    non_slk_bus = (torch.arange(nb) != slk_bus[0]).type(torch.float64)
    slk_vec = torch.zeros_like(non_slk_bus)
    slk_vec[slk_bus[0]] = slk_bus[1]

    R = torch.diag(1.0 / v)
    criterion = WeightedSELoss(z, R)
    optimizer = SGD([T, V], lr=1e-8)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1.5e-1, total_iters=max_iter)

    converged = False
    loss_prev = torch.inf
    for it in tqdm(range(int(max_iter)), desc=f'Optimizing with SGD', leave=False, colour='green'):
        optimizer.zero_grad()
        T_non_slk = T * non_slk_bus + slk_vec
        z_est = h_ac.estimate(T_non_slk, V)
        loss = criterion(z_est)
        loss.backward()
        optimizer.step()
        delta = abs(loss_prev - loss.item())
        if delta < tol:
            converged = True
            break
        loss_prev = loss.item()
        scheduler.step()

        if verbose and it % (max_iter / 100) == 0:
            print(f'SGD - iter: {it}, delta: {delta:.5e}, loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.5e}')
    x = torch.concat([T, V], dim=0)
    return x, T, V, converged, it
