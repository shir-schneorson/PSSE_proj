import torch
import torch_geometric as tg
from sklearn.externals.array_api_compat import device
from torch import nn
import torch.nn.functional as F


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
        self.slk_bus = None

        self.device = kwargs.get('device', 'cpu')
        self.dtype = kwargs.get('dtype', torch.float32)

        self.GNN = GNN_Model(self.k_hops, self.hidden_layers, self.step_dim, self.device, self.dtype).to(self.device)
        self.A = nn.Linear(self.z_dim + self.step_dim,
                           self.v_dim, bias=False).to(device=self.device, dtype=self.dtype)
        self.B = nn.Linear(self.v_dim + self.step_dim,
                           self.v_dim, bias=False).to(device=self.device, dtype=self.dtype)
        self.b = nn.Linear(self.step_dim,
                           self.v_dim, bias=False).to(device=self.device, dtype=self.dtype)
        self.step_emb = nn.Embedding(NUM_STEPS, self.step_dim).to(device=self.device, dtype=self.dtype)

    def forward(self, z, edge_index):
        v0 = torch.zeros(self.batch_size, self.v_dim).to(device=self.device, dtype=self.dtype)
        v = [v0]
        for i in range(self.num_steps):
            step_idx = torch.tensor(i, device=self.device)
            v_i = v[i]
            # X_i_0 = v_i.reshape(-1, 2)
            X_i_0 = v_i.view(self.batch_size, 2, -1).transpose(1, 2).reshape(-1, 2)
            X_i_l = self.GNN(X_i_0, edge_index, step_idx)
            u_i = X_i_l.view(self.batch_size, -1, 2).transpose(1, 2).reshape(self.batch_size, -1)

            s_i = self.step_emb(step_idx)
            s_i = s_i.unsqueeze(0).expand(self.batch_size, -1)

            z_ind = torch.cat([z, s_i], dim=-1)
            u_i_ind = torch.cat([u_i, s_i], dim=-1)

            v_ip1 = self.A(z_ind) + self.B(u_i_ind) + self.b(s_i)
            if self.slk_bus is not None:
                v_ip1[self.slk_bus[0]] = self.slk_bus[1]
            v.append(v_ip1)
        v_last = v[-1]
        return v_last

    def optimize(self, *args):
        z = args[1].to(device=self.device, dtype=self.dtype)
        nb = args[5]
        x0 = torch.zeros(self.v_dim).to(device=self.device, dtype=self.dtype)
        x = [x0]
        for i in range(self.num_steps):
            step_idx = torch.tensor(i, device=self.device)
            x_i = x[i]
            X_i_0 = x_i.reshape(-1, 2)
            X_i_l = self.GNN(X_i_0, self.edge_index, step_idx)
            u_i = X_i_l.T.flatten()

            s_i = self.step_emb(step_idx)

            z_ind = torch.cat([z, s_i], dim=-1)
            u_i_ind = torch.cat([u_i, s_i], dim=-1)
            x_ip1 = self.A(z_ind) + self.B(u_i_ind) + self.b(s_i)
            if self.slk_bus is not None:
                x_ip1[self.slk_bus[0]] = self.slk_bus[1]

            x.append(x_ip1)

        x_last = x[-1]
        T, V = x_last[:nb], x_last[nb:]
        return x_last, T, V, True, self.num_steps


class GNN_Model(nn.Module):
    def __init__(self, k_hops=K_HOPS, hidden_layers=HIDDEN_LAYERS, step_dim=STEP_DIM, device='cpu', dtype=torch.float32):
        super(GNN_Model, self).__init__()
        self.device = device
        self.dtype = dtype
        self.in_channels = 2
        self.out_channels = 2
        self.k_hops = k_hops
        self.hidden_layers = hidden_layers
        self.step_dim = step_dim
        self.step_emb = nn.Embedding(NUM_STEPS, self.step_dim).to(device=self.device, dtype=self.dtype)

        p = list(range(self.k_hops))
        self.mix_hops_layers = nn.ModuleList([
            tg.nn.MixHopConv(in_channels=self.in_channels + self.step_dim,
                             out_channels=self.out_channels,
                             powers=p).to(device=self.device, dtype=self.dtype)
        ])
        for i in range(self.hidden_layers - 1):
            self.mix_hops_layers.append(
                tg.nn.MixHopConv(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 powers=p)).to(device=self.device, dtype=self.dtype)

    def forward(self, X, edge_index, step_index):
        s = self.step_emb(step_index)
        s = s.unsqueeze(0).expand(X.size(0), -1)
        X = torch.cat([X, s], dim=-1)
        for i in range(self.hidden_layers):
            X = self.mix_hops_layers[i](X, edge_index)
            X = X.view(-1, self.k_hops, self.out_channels).sum(dim=1)
            X = F.relu(X)
        return X

# import torch
# import torch_geometric as tg
# from torch import nn
# import torch.nn.functional as F
#
#
# K_HOPS = 2
# HIDDEN_LAYERS = 8
# NUM_STEPS = 10
#
# class GNU_Model(nn.Module):
#     def __init__(self, **kwargs):
#         super(GNU_Model, self).__init__()
#
#         self.z_dim =  kwargs.get('z_dim', 0)
#         self.v_dim = kwargs.get('v_dim', 0)
#
#         if self.z_dim == 0 or self.v_dim == 0:
#             raise ValueError('z_dim and v_dim must be specified and cannot be zero')
#
#         self.k_hops = kwargs.get('k_hops', K_HOPS)
#         self.hidden_layers = kwargs.get('hidden_layers', HIDDEN_LAYERS)
#         self.num_steps = kwargs.get('num_steps', NUM_STEPS)
#         self.edge_index = kwargs.get('edge_index', None)
#
#         self.device = kwargs.get('device', 'cpu')
#         self.dtype = kwargs.get('dtype', torch.get_default_dtype())
#         self.GNN = nn.ModuleList([
#             GNN_Model(self.k_hops, self.hidden_layers, self.device, self.dtype)
#             for _ in range(self.num_steps)
#         ])
#         self.A = nn.ModuleList([
#             nn.Linear(self.z_dim, self.v_dim, bias=False)
#             for _ in range(self.num_steps)
#         ])
#         self.B = nn.ModuleList([
#             nn.Linear(self.v_dim, self.v_dim, bias=False)
#             for _ in range(self.num_steps)
#         ])
#         self.b = nn.ParameterList([
#             nn.Parameter(torch.zeros(self.v_dim))
#             for _ in range(self.num_steps)
#         ])
#
#     def forward(self, z, edge_index):
#         v0 = torch.zeros(self.v_dim).to(device=self.device, dtype=self.dtype)
#         v = [v0]
#         for i in range(self.num_steps):
#             v_i = v[i]
#             X_i_0 = v_i.reshape(-1, 2)
#             X_i_l = self.GNN[i](X_i_0, edge_index)
#             u_i = X_i_l.flatten()
#             v_ip1 = self.A[i](z) + self.B[i](u_i) + self.b[i]
#             v.append(v_ip1)
#         v_last = v[-1]
#         return v_last
#
#     def optimize(self, *args):
#         z = args[1]
#         nb = args[5]
#         x0 = torch.zeros(self.v_dim).to(device=self.device, dtype=self.dtype)
#         x = [x0]
#         for i in range(self.num_steps):
#             x_i = x[i]
#             X_i_0 = x_i.reshape(-1, 2)
#             X_i_l = self.GNN[i](X_i_0, self.edge_index)
#             u_i = X_i_l.flatten()
#             x_ip1 = self.A[i](z) + self.B[i](u_i) + self.b[i]
#             x.append(x_ip1)
#         x_last = x[-1]
#         T, V = x_last[:nb], x_last[nb:]
#         return x_last, T, V, True, self.num_steps
#
#
# class GNN_Model(nn.Module):
#     def __init__(self, k_hops=K_HOPS, hidden_layers=HIDDEN_LAYERS, device='cpu', dtype=torch.float32):
#         super(GNN_Model, self).__init__()
#         self.device = device
#         self.dtype = dtype
#         self.in_channels = 2
#         self.out_channels = 2
#         self.k_hops = k_hops
#         self.hidden_layers = hidden_layers
#
#         p = list(range(self.k_hops))
#         self.mix_hops_layers = nn.ModuleList([
#             tg.nn.MixHopConv(in_channels=self.in_channels, out_channels=self.out_channels, powers=p)
#             for _ in range(self.hidden_layers)
#         ])
#
#     def forward(self, X, edge_index):
#         for i in range(self.hidden_layers):
#             X = self.mix_hops_layers[i](X, edge_index)
#             X = X.view(-1, self.k_hops, self.out_channels).sum(dim=1)
#             X = F.relu(X)
#
#         return X
