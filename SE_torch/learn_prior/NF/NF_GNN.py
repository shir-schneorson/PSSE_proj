# ================================================================
# One copy-paste: Graph-conditioned RealNVP flow (GNN conditioner)
# Replaces MLP affine coupling with GNN-based conditioner using edge_index.
#
# Requirements:
#   pip install torch-geometric (matching your torch version)
#   You must provide:
#     - nb: number of buses (nodes)
#     - edge_index: LongTensor [2, E] (PyG format, 0-based)
#     - data y: Tensor [B, 2*nb] where each node has 2 features (e.g., [theta_i, V_i])
#
# This is an invertible normalizing flow: z = f^{-1}(y), y = f(z)
# Loss: E[-log p(z) - log|det J_{f^{-1}}(y)|]
# ================================================================

import os
import json
import math as mt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm.auto import tqdm

import torch_geometric as tg

from SE_torch.learn_prior.GNU_GNN.load_data import load_data


# -----------------------
# Utils: vec <-> nodes
# -----------------------
def vec_to_nodes(y: torch.Tensor, nb: int) -> torch.Tensor:
    # y: [B, 2*nb] -> X: [B, nb, 2]
    B = y.shape[0]
    return y.view(B, nb, 2)


def nodes_to_vec(X: torch.Tensor) -> torch.Tensor:
    # X: [B, nb, 2] -> y: [B, 2*nb]
    return X.reshape(X.shape[0], -1)


def make_alternating_node_masks(nb: int, num_blocks: int, device=None) -> list[torch.Tensor]:
    # mask=1 means "conditioned" nodes (unchanged); mask=0 means "transformed" nodes
    base = (torch.arange(nb, device=device) % 2).float()  # 0,1,0,1...
    masks = []
    for i in range(num_blocks):
        masks.append(base if i % 2 == 0 else 1.0 - base)
    return masks


def batch_edge_index(edge_index: torch.Tensor, nb: int, B: int) -> torch.Tensor:
    """
    edge_index: [2, E] for single graph with nodes 0..nb-1
    Returns edge_index_batched: [2, B*E] for B disjoint copies.
    """
    E = edge_index.shape[1]
    edge_index = edge_index.to(dtype=torch.long)
    offsets = (torch.arange(B, device=edge_index.device) * nb).repeat_interleave(E)  # [B*E]
    return edge_index.repeat(1, B) + offsets.unsqueeze(0)


# -----------------------
# Permutation (kept)
# -----------------------
class PermutationalLayer(nn.Module):
    def __init__(self, data_dim: int):
        super().__init__()
        perm = torch.randperm(data_dim)
        while torch.equal(perm, torch.arange(data_dim)):
            perm = torch.randperm(data_dim)
        inv_perm = torch.argsort(perm)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, self.perm]

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y[:, self.inv_perm]


# -----------------------
# GNN conditioner
# -----------------------
class GNNConditioner(nn.Module):
    """
    Input:  X_masked [B, nb, 2]
    Output: log_s [B, nb, 2], b [B, nb, 2]
    """
    def __init__(self, nb: int, hidden_dim: int = 64, k_hops: int = 2, S_MAX: float = 1.0):
        super().__init__()
        self.nb = nb
        self.S_MAX = S_MAX
        powers = list(range(k_hops))

        # MixHopConv returns [N, len(powers)*out_channels]
        self.conv1 = tg.nn.MixHopConv(in_channels=2, out_channels=hidden_dim, powers=powers)
        self.conv2 = tg.nn.MixHopConv(in_channels=hidden_dim, out_channels=hidden_dim, powers=powers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),  # 2 for log_s, 2 for b
        )

    def _collapse_mixhop(self, H: torch.Tensor, k_hops: int) -> torch.Tensor:
        # H: [N, k_hops*out_channels] -> [N, out_channels]
        return H.view(H.shape[0], k_hops, -1).sum(dim=1)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, nb, C = X.shape
        assert nb == self.nb and C == 2

        Xg = X.reshape(B * nb, C)                       # [B*nb, 2]
        eib = batch_edge_index(edge_index, nb, B)       # [2, B*E]

        H = self.conv1(Xg, eib)                         # [B*nb, k_hops*hidden_dim]
        k_hops = self.conv1.powers.__len__() if hasattr(self.conv1, "powers") else 2
        H = self._collapse_mixhop(H, k_hops)            # [B*nb, hidden_dim]
        H = F.silu(H)

        H = self.conv2(H, eib)
        k_hops2 = self.conv2.powers.__len__() if hasattr(self.conv2, "powers") else 2
        H = self._collapse_mixhop(H, k_hops2)
        H = F.silu(H)

        out = self.head(H).view(B, nb, 4)
        log_s = out[..., :2]
        b = out[..., 2:]

        log_s = self.S_MAX * torch.tanh(log_s)
        return log_s, b


# -----------------------
# Graph affine coupling (invertible)
# -----------------------
class GraphAffineCouplingLayer(nn.Module):
    def __init__(self, nb: int, node_mask: torch.Tensor, conditioner: GNNConditioner):
        super().__init__()
        self.nb = nb
        self.conditioner = conditioner
        assert node_mask.shape == (nb,)
        self.register_buffer("node_mask", node_mask.float().view(1, nb, 1))  # [1,nb,1]

    def forward(self, y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # y: [B, 2*nb]
        X = vec_to_nodes(y, self.nb)     # [B,nb,2]
        Xm = X * self.node_mask          # conditioned nodes
        log_s, b = self.conditioner(Xm, edge_index)
        s = torch.exp(log_s)

        inv_mask = 1.0 - self.node_mask  # transformed nodes
        Y = Xm + inv_mask * (s * (X * inv_mask) + b)
        return nodes_to_vec(Y)

    def inverse(self, y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        X = vec_to_nodes(y, self.nb)
        Xm = X * self.node_mask
        log_s, b = self.conditioner(Xm, edge_index)
        s = torch.exp(log_s)

        inv_mask = 1.0 - self.node_mask
        Z = Xm + inv_mask * ((X * inv_mask - b) / (s + 1e-12))
        return nodes_to_vec(Z)

    def log_det_inv_jacobian(self, y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # log|det d(inv)/dy| = - sum log_s over transformed dims
        X = vec_to_nodes(y, self.nb)
        Xm = X * self.node_mask
        log_s, _ = self.conditioner(Xm, edge_index)
        inv_mask = 1.0 - self.node_mask
        return torch.sum(-log_s * inv_mask, dim=(1, 2))  # [B]


# -----------------------
# Flow model using GNN-conditioned couplings
# -----------------------
class FlowModelGNN(nn.Module):
    def __init__(self, **kwargs): #nb: int, num_blocks: int = 4, hidden_dim: int = 64, k_hops: int = 2, S_MAX: float = 1.0
        super().__init__()
        self.nb = kwargs.get('nb', 118)
        self.data_dim = 2 * self.nb
        self.num_blocks = kwargs.get('num_blocks', 4)
        self.hidden_dim = kwargs.get('hidden_dim', 64)
        self.k_hops = kwargs.get('k_hops', 2)
        self.S_MAX = kwargs.get('S_MAX', 1.0)

        self.perm_layers = nn.ModuleList([PermutationalLayer(self.data_dim) for _ in range(self.num_blocks)])

        masks = make_alternating_node_masks(self.nb, self.num_blocks)
        self.coupling_layers = nn.ModuleList([])
        for i in range(self.num_blocks):
            cond = GNNConditioner(nb=self.nb, hidden_dim=self.hidden_dim, k_hops=self.k_hops, S_MAX=self.S_MAX)
            self.coupling_layers.append(GraphAffineCouplingLayer(self.nb, masks[i], cond))

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for coup, perm in zip(self.coupling_layers, self.perm_layers):
            z = coup(perm(z), edge_index)
        return z

    def inverse(self, y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for coup, perm in reversed(list(zip(self.coupling_layers, self.perm_layers))):
            y = perm.inverse(coup.inverse(y, edge_index))
        return y

    def log_det_inv_jacobian(self, y: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        log_det = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        for coup, perm in reversed(list(zip(self.coupling_layers, self.perm_layers))):
            log_det = log_det + coup.log_det_inv_jacobian(y, edge_index)
            y = perm.inverse(coup.inverse(y, edge_index))
        return log_det


# -----------------------
# Trainer (edge_index-aware)
# -----------------------
class NormalizingFlowTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        test_loader,
        num_epochs: int = 20,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        betas=(0.9, 0.99),
        ckpt_path: str = "",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.ckpt_path = ckpt_path
        self.device = device
        self.dtype = dtype

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        # Standard normal prior in latent space z
        data_dim = model.data_dim
        mean = torch.zeros(data_dim, device=device, dtype=dtype)
        cov = torch.eye(data_dim, device=device, dtype=dtype)
        self.prior_dist = MultivariateNormal(mean, cov)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running = 0.0
        desc = f"{self.model.__class__.__name__} [Epoch {epoch}] 🟢 Training"
        pbar = tqdm(self.train_loader, desc=desc, colour="green")
        for batch in pbar:
            y, edge_index = batch.x, batch.edge_index
            y = y.view(batch.batch_size, -1, 2).transpose(1, 2).reshape(batch.batch_size, -1)
            edge_index = edge_index.T.view(batch.batch_size, -1, 2)[0].T
            y = y.to(device=self.device, dtype=self.dtype)
            edge_index = edge_index.to(device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            z = self.model.inverse(y, edge_index)
            log_prob = self.prior_dist.log_prob(z)  # [B]
            log_det = self.model.log_det_inv_jacobian(y, edge_index)  # [B]
            loss = torch.mean(-(log_prob + log_det))

            loss.backward()
            self.optimizer.step()
            running += float(loss.item())
            pbar.set_postfix(loss=loss.item())

        self.scheduler.step()
        return running / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate_epoch(self, epoch: int):
        self.model.eval()
        agg_loss = agg_logdet = agg_logprob = 0.0
        desc = f"{self.model.__class__.__name__} [Epoch {epoch}] 🔵 Validating"

        for batch in tqdm(self.test_loader, desc=desc, colour="blue"):
            y, edge_index = batch.x, batch.edge_index
            y = y.view(batch.batch_size, -1, 2).transpose(1, 2).reshape(batch.batch_size, -1)
            edge_index = edge_index.T.view(batch.batch_size, -1, 2)[0].T
            y = y.to(device=self.device, dtype=self.dtype)
            edge_index = edge_index.to(device=self.device)
            z = self.model.inverse(y, edge_index)
            log_prob = self.prior_dist.log_prob(z)
            log_det = self.model.log_det_inv_jacobian(y, edge_index)
            loss = torch.mean(-(log_prob + log_det))

            agg_loss += float(loss.item())
            agg_logdet += float(torch.mean(log_det).item())
            agg_logprob += float(torch.mean(log_prob).item())

        n = max(1, len(self.test_loader))
        return agg_loss / n, agg_logdet / n, agg_logprob / n

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            tr = self.train_epoch(epoch)
            va, ld, lp = self.validate_epoch(epoch)
            print(f"Epoch {epoch:03d} | Train {tr:.4f} | Val {va:.4f} | LogDet {ld:.4f} | LogProb {lp:.4f}")

        if self.ckpt_path:
            os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.ckpt_path)


def train_normalizing_flow(config_path):
    config = json.load(open(config_path))
    device = config.get('device', 'mps')
    dtype = config.get('dtype', torch.get_default_dtype())
    num_samples = config.get('num_samples', 100000)

    train_loader, test_loader, train_dataset, test_dataset, config = load_data(config, num_samples)

    model = FlowModelGNN(**config).to(device=device, dtype=dtype)

    # Load weights if present
    ckpt_path = f"./models/{config.get('ckpt_name')}"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    trainer = NormalizingFlowTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=20,
        learning_rate=1e-2,
        weight_decay=1e-3,
        ckpt_path=ckpt_path,
        device=device,
        dtype=dtype,
    )
    trainer.train()


if __name__ == "__main__":
    # for config_name in os.listdir("../configs"):
    config_name = "NF_4_0_235.json"
    config_path = f'../configs/{config_name}'
    train_normalizing_flow(config_path)