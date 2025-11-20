import torch
from torch import nn

K_HOPS = 2
HIDDEN_LAYERS = 8
NUM_STEPS = 10

class GNU_Model(nn.Module):
    def __init__(self, **kwargs):
        super(GNU_Model, self).__init__()

        self.z_dim =  kwargs.get('z_dim', 0)
        self.v_dim = kwargs.get('v_dim', 0)

        if self.z_dim == 0 or self.v_dim == 0:
            raise ValueError('z_dim and v_dim must be specified and cannot be zero')

        self.k_hops = kwargs.get('k_hops', K_HOPS)
        self.hidden_layers = kwargs.get('hidden_layers', HIDDEN_LAYERS)
        self.num_steps = kwargs.get('num_steps', NUM_STEPS)

        self.device = kwargs.get('device', 'cpu')
        self.dtype = kwargs.get('dtype', torch.get_default_dtype())

        self.GNN = [GNN_Model(self.k_hops, self.hidden_layers, self.device, self.dtype) for _ in range(self.num_steps)]
        self.A = [nn.Linear(self.z_dim, self.v_dim, bias=False) for _ in range(self.num_steps)]
        self.B = [nn.Linear(self.v_dim, self.v_dim, bias=False) for _ in range(self.num_steps)]
        self.b = [nn.Parameter(torch.zeros(self.v_dim)) for _ in range(self.num_steps)]

    def forward(self, z):
        v0 = torch.zeros(self.v_dim).to(device=self.device, dtype=self.dtype)
        v = [v0]
        for i in range(self.num_steps):
            v_i = v[i]
            X_i_0 = v_i.reshape(-1, 2)
            X_i_l = self.GNN[i](X_i_0)
            u_i = X_i_l.flatten()
            v_ip1 = self.A[i](z) + self.B[i](u_i) + self.b[i]
            v.append(v_ip1)
        v_last = v[-1]
        return v_last


class GNN_Model(nn.Module):
    def __init__(self, k_hops=K_HOPS, hidden_layers=HIDDEN_LAYERS, device='cpu', dtype=torch.float32):
        super(GNN_Model, self).__init__()
        self.device = device
        self.dtype = dtype
        self.k_hops = k_hops
        self.hidden_layers = hidden_layers
