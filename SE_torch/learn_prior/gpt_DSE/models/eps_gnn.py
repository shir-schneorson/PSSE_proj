
import torch
import torch.nn as nn
from .layers import mlp, MessagePassingBlock

class EpsGNN(nn.Module):
    def __init__(self, node_in=3, node_meta_dim=4, edge_in=4, hidden=128, layers=6, time_dim=128, out_dim=3):
        super().__init__()
        self.enc_node = mlp(node_in + node_meta_dim, hidden, hidden, depth=2)
        self.enc_edge = mlp(edge_in, hidden, hidden, depth=2)
        self.enc_time = mlp(time_dim, hidden, hidden, depth=2)
        self.blocks = nn.ModuleList([MessagePassingBlock(hidden) for _ in range(layers)])
        self.dec_node = mlp(hidden, out_dim, hidden, depth=2)

    def forward(self, x_t, node_meta, edge_index, edge_feats, t_embed, batch_index):
        h_node = self.enc_node(torch.cat([x_t, node_meta], dim=-1)) + self.enc_time(t_embed)
        h_edge = self.enc_edge(edge_feats)
        for blk in self.blocks:
            h_node = blk(h_node, h_edge, edge_index)
        eps_hat = self.dec_node(h_node)
        return eps_hat
