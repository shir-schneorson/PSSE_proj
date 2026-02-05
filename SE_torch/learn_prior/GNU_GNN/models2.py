import torch
import torch_geometric as tg
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

K_HOPS = 2
HIDDEN_LAYERS = 8
NUM_STEPS = 100
STEP_DIM = 8

class GNU_Model(nn.Module):
    def __init__(self, **kwargs):
        super(GNU_Model, self).__init__()

        self.z_dim =  kwargs.get('z_dim', 0)
        self.v_dim = kwargs.get('v_dim', 0)
        self.batch_size = kwargs.get('batch_size', 1)

        if self.z_dim == 0 or self.v_dim == 0:
            raise ValueError('z_dim and v_dim must be specified and cannot be zero')

        self.k_hops = kwargs.get('k_hops', K_HOPS)
        self.hidden_layers = kwargs.get('hidden_layers', HIDDEN_LAYERS)
        self.num_steps = kwargs.get('num_steps', NUM_STEPS)
        self.step_dim = kwargs.get('step_dim', STEP_DIM)
        self.edge_index = kwargs.get('edge_index', None)
        self.slk_bus = kwargs.get('slk_bus', None)

        self.device = kwargs.get('device', 'cpu')
        self.dtype = kwargs.get('dtype', torch.float32)
        self.GNN = nn.ModuleList([
                        GNN_Model(self.k_hops, self.hidden_layers, self.device, self.dtype)
                        for _ in range(self.num_steps)
                    ])
        self.A = nn.ModuleList([
            nn.Linear(self.z_dim, self.v_dim, bias=False)
            for _ in range(self.num_steps)
        ])
        self.B = nn.ModuleList([
            nn.Linear(self.v_dim, self.v_dim, bias=False)
            for _ in range(self.num_steps)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(self.v_dim))
            for _ in range(self.num_steps)
        ])

    def forward(self, z, edge_index):
        bs = z.size(0)
        v0 = torch.cat([torch.zeros(bs, self.v_dim // 2),
                        torch.ones(bs, self.v_dim // 2)], dim=1).to(device=self.device, dtype=self.dtype)
        v0.requires_grad = True
        v = [v0]
        for i in range(self.num_steps):
            v_i = v[i]
            X_i_0 = v_i.view(bs, 2, -1).transpose(1, 2).reshape(-1, 2)
            X_i_l = self.GNN[i](X_i_0, edge_index)
            u_i = X_i_l.view(bs, -1, 2).transpose(1, 2).reshape(bs, -1)
            v_ip1 = self.A[i](z) + self.B[i](u_i) + self.b[i]
            if self.slk_bus is not None:
                idx = torch.tensor([int(self.slk_bus[0])], device=v_ip1.device)
                val = torch.tensor([float(self.slk_bus[1])], device=v_ip1.device, dtype=v_ip1.dtype)
                v_ip1 = v_ip1.index_copy(1, idx, val.expand(bs, 1))
            v.append(v_ip1)
        v_last = v[-1]
        return v_last, v

    def optimize(self, *args):
        z = args[1].to(device=self.device, dtype=self.dtype)
        z = z.view(1, -1)
        nb = args[5]
        bs = z.size(0)
        x0 = torch.cat([torch.zeros(bs, self.v_dim // 2), torch.ones(bs, self.v_dim // 2)], dim=1).to(
            device=self.device, dtype=self.dtype)
        all_x = [x0.clone().detach().flatten()]
        pbar = tqdm(range(self.num_steps), desc="Optimizing GNU GNN", leave=True)
        for i in pbar:
            x_i = all_x[i]
            X_i_0 = x_i.view(bs, 2, -1).transpose(1, 2).reshape(-1, 2)
            X_i_l = self.GNN[i](X_i_0, self.edge_index)
            u_i = X_i_l.view(bs, -1, 2).transpose(1, 2).reshape(bs, -1)
            x_ip1 = self.A[i](z) + self.B[i](u_i) + self.b[i]
            if self.slk_bus is not None:
                x_ip1[:, self.slk_bus[0]] = self.slk_bus[1]
            all_x.append(x_ip1.clone().detach().flatten())

        x_last = all_x[-1].flatten()
        T, V = x_last[:nb], x_last[nb:]
        return x_last, T, V, True, self.num_steps, torch.nan, all_x


class GNN_Model(nn.Module):
    def __init__(self, k_hops=K_HOPS, hidden_layers=HIDDEN_LAYERS, device='cpu', dtype=torch.float32):
        super(GNN_Model, self).__init__()
        self.device = device
        self.dtype = dtype
        self.in_channels = 2
        self.out_channels = 2
        self.k_hops = k_hops
        self.hidden_layers = hidden_layers

        p = list(range(self.k_hops))
        self.mix_hops_layers = nn.ModuleList([
            tg.nn.MixHopConv(in_channels=self.in_channels, out_channels=self.out_channels, powers=p)
            for _ in range(self.hidden_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(self.out_channels) for _ in range(self.hidden_layers)])

    def forward(self, X, edge_index):
        for i in range(self.hidden_layers):
            X = self.mix_hops_layers[i](X, edge_index)
            X = X.view(-1, self.k_hops, self.out_channels).sum(dim=1)
            X = self.norms[i](X)
            X = F.leaky_relu(X, 0.01)
        return X
