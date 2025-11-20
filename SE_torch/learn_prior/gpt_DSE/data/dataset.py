import os
import torch
from torch.utils.data import Dataset

from SE_torch.data_generator import DataGenerator
from SE_torch.learn_prior.gpt_DSE.data.scalers import ChannelScaler
from SE_torch.utils import init_start_point

float_dtype = torch.float32
torch.set_default_dtype(float_dtype)


def one_hot(idx, num_classes):
    v = torch.zeros(num_classes, dtype=float_dtype)
    v[idx] = 1.0
    return v

class GraphSample:
    """Lightweight container for a single graph sample (one operating point)."""
    def __init__(self, node_feats, node_meta, edge_index, edge_feats, slack_mask, batch_index):
        self.node_feats = node_feats      # [N, 3]      -> [V, sinθ, cosθ] (normalized)
        self.node_meta  = node_meta       # [N, p]
        self.edge_index = edge_index      # [2, E] (long)
        self.edge_feats = edge_feats      # [E, q]
        self.slack_mask = slack_mask      # [N, 3] 1=learn; 0=freeze (e.g., sin/cos at slack)
        self.batch_index= batch_index     # [N] all zeros (single-graph)

def sys_to_graph(sys, x0=None):
    N = sys.nb
    bus_df = sys.bus.sort_values('idx_bus').reset_index(drop=True)

    node_meta_cols = ['rsh', 'xsh', 'Vmin', 'Vmax', 'Qmin', 'Qmax']
    node_meta_num = torch.as_tensor(bus_df[node_meta_cols].to_numpy(), dtype=float_dtype)

    bus_type = torch.as_tensor(bus_df['bus_type'].to_numpy(), dtype=torch.long)
    node_meta_type = torch.stack([one_hot(int(bt) - 1, 3) for bt in bus_type], dim=0)

    node_meta = torch.cat([node_meta_num, node_meta_type], dim=1)
    p = node_meta.shape[1]

    br = sys.branch
    idx_from = torch.as_tensor(br['idx_from'].to_numpy(), dtype=torch.long)
    idx_to = torch.as_tensor(br['idx_to'].to_numpy(), dtype=torch.long)
    edge_index = torch.stack([idx_from, idx_to], dim=0)

    edge_feats_cols = ['rij', 'xij', 'bsi', 'tij', 'fij']
    edge_feats = torch.as_tensor(br[edge_feats_cols].to_numpy(), dtype=float_dtype)
    q = edge_feats.shape[1]

    slack_mask = torch.ones((N, 3), dtype=float_dtype)
    slack_nodes = (bus_type == 3).nonzero(as_tuple=True)[0]
    if slack_nodes.numel() > 0:
        slack_mask[slack_nodes, 1:] = 0.0

    batch_index = torch.zeros(N, dtype=torch.long)

    if x0 is not None:
        T = x0[:N]
        V = x0[N:]
    else:
        T, V = init_start_point(sys, how='flat')

    V = V.unsqueeze(-1)
    sinT = torch.sin(T).unsqueeze(-1)
    cosT = torch.cos(T).unsqueeze(-1)
    node_feats = torch.cat([V, sinT, cosT], dim=-1)

    return node_feats, node_meta, edge_index, edge_feats, slack_mask, batch_index



class GraphDataset(Dataset):
    def __init__(self, sys, n_samples: int, path: str = '../../datasets/data.pt'):
        super().__init__()
        self.path = path
        self.sys = sys
        self.n_samples = n_samples
        self.samples = self._load_samples()

    def _load_samples(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if os.path.exists(self.path):
            data = torch.load(self.path).to(float_dtype)
            # sanity: if file contains dict with 'data', handle it
            if isinstance(data, dict) and 'data' in data:
                data = data['data'].to(float_dtype)
        else:
            data_generator = DataGenerator()
            T, V = data_generator.sample(self.sys, num_samples=self.n_samples, random_flow=True)
            T = torch.as_tensor(T, dtype=float_dtype)
            V = torch.as_tensor(V, dtype=float_dtype)
            data = torch.cat([T, V], dim=1)  # [S, 2N]
            torch.save(data, self.path)

        S, twoN = data.shape
        N = self.sys.nb

        bus_df = self.sys.bus.sort_values('idx_bus').reset_index(drop=True)

        node_meta_cols = ['rsh','xsh','Vmin','Vmax', 'Qmin','Qmax']
        node_meta_num = torch.as_tensor(bus_df[node_meta_cols].to_numpy(), dtype=float_dtype)

        bus_type = torch.as_tensor(bus_df['bus_type'].to_numpy(), dtype=torch.long)
        node_meta_type = torch.stack([one_hot(int(bt) - 1, 3) for bt in bus_type], dim=0)

        node_meta = torch.cat([node_meta_num, node_meta_type], dim=1)
        p = node_meta.shape[1]

        br = self.sys.branch
        idx_from = torch.as_tensor(br['idx_from'].to_numpy(), dtype=torch.long)
        idx_to   = torch.as_tensor(br['idx_to'  ].to_numpy(), dtype=torch.long)
        edge_index = torch.stack([idx_from, idx_to], dim=0)

        edge_feats_cols = ['rij','xij','bsi','tij','fij']
        edge_feats = torch.as_tensor(br[edge_feats_cols].to_numpy(), dtype=float_dtype)
        q = edge_feats.shape[1]

        slack_mask = torch.ones((N, 2), dtype=float_dtype)
        slack_nodes = (bus_type == 3).nonzero(as_tuple=True)[0]
        if slack_nodes.numel() > 0:
            slack_mask[slack_nodes, 0] = 0.0

        batch_index = torch.zeros(N, dtype=torch.long)

        T_all = data[:, :N]
        V_all = data[:, N:]

        T = T_all.unsqueeze(-1)
        V = V_all.unsqueeze(-1)
        # sinT = torch.sin(T_all).unsqueeze(-1)
        # cosT = torch.cos(T_all).unsqueeze(-1)
        # node_feats_all = torch.cat([V3, sinT, cosT], dim=-1)
        node_feats_all = torch.cat([T, V], dim=-1)
        scaler = ChannelScaler()
        scaler.fit(node_feats_all)
        node_feats_all_norm = scaler.transform(node_feats_all)
        scaler.save('../../datasets/scaler')
        # ---- 4) Normalize node_feats per channel over all nodes+samples ----
        # shape [S*N, 3]
        # flat = node_feats_all.reshape(-1, 2)
        # mean = node_feats_all.mean(dim=0)                           # [3]
        # std  = node_feats_all.std(dim=0)
        # std[std < 1e-10] = 1
        # node_feats_all_norm = (node_feats_all - mean) / std

        # Save scalers on the dataset (useful later for inverse)
        # self.node_feats_mean = mean
        # self.node_feats_std  = std
        self.node_meta_dim = p
        self.edge_feat_dim = q

        samples = []
        for s in range(S):
            node_feats = node_feats_all_norm[s]
            samples.append(
                GraphSample(
                    node_feats=node_feats.clone(),
                    node_meta=node_meta.clone(),
                    edge_index=edge_index.clone(),
                    edge_feats=edge_feats.clone(),
                    slack_mask=slack_mask.clone(),
                    batch_index=batch_index.clone()
                )
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return {
            "node_feats":  self.samples[idx].node_feats,
            "node_meta":   self.samples[idx].node_meta,
            "edge_index":  self.samples[idx].edge_index,
            "edge_feats":  self.samples[idx].edge_feats,
            "slack_mask":  self.samples[idx].slack_mask,
            "batch_index": self.samples[idx].batch_index,
        }

def collate_fn(batch):
    """
    Concatenate a list of GraphSample dicts into a single mini-batch.
    PyG-style:
      - node tensors are concatenated along dim 0
      - edge_index is remapped with running node offsets
      - batch_index marks which graph each node belongs to (0..B-1)
    """

    # Storage
    node_feats_list   = []
    node_meta_list    = []
    edge_index_list   = []
    edge_feats_list   = []
    slack_mask_list   = []
    batch_index_list  = []   # rebuilt (ignore per-sample placeholders)

    node_offset = 0
    for g_id, item in enumerate(batch):
        # Required tensors per graph
        x = item["node_feats"]      # [N_i, 3], float64
        x_meta = item["node_meta"]       # [N_i, p], float64
        ei = item["edge_index"]      # [2, E_i], long
        ef = item["edge_feats"]      # [E_i, q], float64
        smask = item["slack_mask"]      # [N_i, 3], float64

        N_i = x.shape[0]
        # Remap edges by node offset
        ei_remap = ei + node_offset

        # Batch index for these N_i nodes
        bi = torch.full((N_i,), g_id, dtype=torch.long, device=x.device)

        # Append
        node_feats_list.append(x)
        node_meta_list.append(x_meta)
        edge_index_list.append(ei_remap)
        edge_feats_list.append(ef)
        slack_mask_list.append(smask)
        batch_index_list.append(bi)

        node_offset += N_i

    # Concatenate
    node_feats  = torch.cat(node_feats_list,  dim=0)                    # [ΣN, 3]
    node_meta   = torch.cat(node_meta_list,   dim=0)                    # [ΣN, p]
    edge_index  = torch.cat(edge_index_list,  dim=1)                    # [2, ΣE]
    edge_feats  = torch.cat(edge_feats_list,  dim=0)                    # [ΣE, q]
    slack_mask  = torch.cat(slack_mask_list,  dim=0)                    # [ΣN, 3]
    batch_index = torch.cat(batch_index_list, dim=0)                    # [ΣN]

    # Ensure dtypes (match your training to float64 / long)
    node_feats  = node_feats.to(float_dtype)
    node_meta   = node_meta.to(float_dtype)
    edge_feats  = edge_feats.to(float_dtype)
    slack_mask  = slack_mask.to(float_dtype)
    edge_index  = edge_index.to(torch.long)
    batch_index = batch_index.to(torch.long)

    return {
        "node_feats":  node_feats,
        "node_meta":   node_meta,
        "edge_index":  edge_index,
        "edge_feats":  edge_feats,
        "slack_mask":  slack_mask,
        "batch_index": batch_index,
    }
