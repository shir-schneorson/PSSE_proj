
import torch
import torch.nn as nn

def mlp(in_dim, out_dim, hidden=128, depth=2, act=nn.SiLU):
    layers = []
    dims = [in_dim] + [hidden]*(depth-1) + [out_dim]
    for i in range(len(dims)-2):
        layers += [nn.Linear(dims[i], dims[i+1], dtype=torch.float32), act()]
    layers += [nn.Linear(dims[-2], dims[-1], torch.float32)]
    return nn.Sequential(*layers)

class MessagePassingBlock(nn.Module):
    """Placeholder message-passing block.
    Replace with torch_geometric.nn modules (e.g., GAT, GINE) in your implementation.
    """
    def __init__(self, hidden):
        super().__init__()
        self.msg_mlp = mlp(3*hidden, hidden, hidden, depth=2)
        self.upd_mlp = mlp(2*hidden, hidden, hidden, depth=2)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h_node, h_edge, edge_index):
        # edge_index: [2, E] with src->dst edges
        src, dst = edge_index
        # naive gather (no PyG); this is O(E) indexing and not memory efficient; replace with PyG
        m_in = torch.cat([h_node[src], h_node[dst], h_edge], dim=-1)
        m = self.msg_mlp(m_in)

        # aggregate to dst by sum
        agg = torch.zeros_like(h_node)
        agg.index_add_(0, dst, m)

        h_new = self.upd_mlp(torch.cat([h_node, agg], dim=-1))
        return self.norm(h_node + h_new)
