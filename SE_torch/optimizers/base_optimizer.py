import json

import torch
from torch import nn as nn
from torch.distributions import MultivariateNormal

from SE_torch.learn_prior.NF import FlowModel
from SE_torch.learn_prior.NF import DATA_DIM

torch.set_default_dtype(torch.float64)

class SEOptimizer:
    def __init__(self, **kwargs):
        self.tol = kwargs.get('tol', 1e-10)
        self.max_iter = kwargs.get('max_iter', 100)

    def __call__(self, x0, z, v, slk_bus, h_ac, nb):
        raise NotImplementedError


class WeightedSELoss(nn.Module):
    def __init__(self, z, R, norm_H=None):
        super().__init__()
        self.z = z
        self.R = R
        self.norm_H = norm_H if norm_H is not None else torch.ones_like(z)

    def forward(self, z_est):
        error = ((self.z - z_est) / self.norm_H).reshape(-1, 1)
        weighted_squared_error = (error.T @ self.R @ error)
        return weighted_squared_error

class WeightedSEWithPriorLoss(nn.Module):
    def __init__(self, z, R, slk_bus, NF_config_path='../learn_prior/configs/NF_2_0_64.json', norm_H=None):
        super().__init__()
        self.z = z
        self.R = R
        self.slk_bus = slk_bus
        NF_config =  json.load(open(NF_config_path))
        ckpt_path = f"../learn_prior/models/{NF_config.get('ckpt_name')}"
        self.flow_model = FlowModel(**NF_config)
        self.flow_model.load_state_dict(torch.load(ckpt_path))
        self.mean = torch.load("../learn_prior/datasets/mean.pt").to(torch.float64)
        self.std = torch.load("../learn_prior/datasets/std.pt").to(torch.float64)
        self.prior_dist = MultivariateNormal(torch.zeros(DATA_DIM), torch.eye(DATA_DIM))
        self.norm_H = norm_H if norm_H is not None else torch.ones_like(z)
        self.lamda_iter =  iter(torch.linspace(1, 1e-2, steps=50))

    def forward(self, z_est, x, lamda):
        self.flow_model.eval()
        x = torch.concat([x[:self.slk_bus], x[self.slk_bus + 1:]])#.to(dtype=torch.float32)
        x_norm = ((x - self.mean) /self.std).unsqueeze(0).to(device='cpu', non_blocking=False)
        z = self.flow_model.inverse(x_norm).to(device='cpu')
        log_prob = self.prior_dist.log_prob(z)
        log_det_inv_jacobian = self.flow_model.log_det_inv_jacobian(x_norm).to(device='cpu')
        loss_prior = torch.mean(-1 * (log_prob + log_det_inv_jacobian))
        error = ((self.z - z_est) / self.norm_H).reshape(-1, 1)
        loss_meas = (error.T @ self.R @ error)
        loss = loss_meas + lamda * loss_prior
        return loss