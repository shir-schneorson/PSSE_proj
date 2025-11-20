
import torch
from models.eps_gnn import EpsGNN
from utils.schedules import make_schedule
from utils.time_embed import FourierTimeEmbedding

@torch.no_grad()
def ddim_update(x, eps_hat, t, t_prev, alpha_bars):
    # Simple DDIM-like deterministic update
    ab_t = alpha_bars[t]
    ab_s = alpha_bars[t_prev]
    sigma_t = torch.sqrt(1.0 - ab_t).unsqueeze(-1)
    pred_x0 = (x - sigma_t * eps_hat) / torch.sqrt(ab_t).unsqueeze(-1)
    x = torch.sqrt(ab_s).unsqueeze(-1) * pred_x0 + torch.sqrt(1.0 - ab_s).unsqueeze(-1) * eps_hat
    return x

def project_slack(x, slack_mask):
    return x * slack_mask + (1 - slack_mask) * x.detach()

@torch.no_grad()
def sample_prior(graph_template, ckpt_path, cfg, n_steps=50, ode=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = make_schedule(T, cfg["diffusion"]["schedule"])
    betas, alphas, alpha_bars = betas.to(device), alphas.to(device), alpha_bars.to(device)

    model = EpsGNN(**cfg["model"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    time_embed = FourierTimeEmbedding(cfg["diffusion"]["time_embed_dim"]).to(device)

    x = torch.randn_like(graph_template["node_feats"]).to(device)
    node_meta = graph_template["node_meta"].to(device)
    edge_index = graph_template["edge_index"].to(device)
    edge_feats = graph_template["edge_feats"].to(device)
    slack_mask = graph_template["slack_mask"].to(device)
    batch_index = graph_template["batch_index"].to(device)

    # Uniform coarse schedule
    ts = torch.linspace(T-1, 0, steps=n_steps, dtype=torch.long, device=device)

    for i in range(len(ts)-1):
        t = ts[i]
        t_prev = ts[i+1]
        t_nodes = t.expand_as(batch_index)
        t_emb = time_embed(t_nodes, T)

        eps_hat = model(x, node_meta, edge_index, edge_feats, t_emb, batch_index)

        if ode:
            x = ddim_update(x, eps_hat, t, t_prev, alpha_bars)
        else:
            # SDE-style step (add small noise if desired)
            x = ddim_update(x, eps_hat, t, t_prev, alpha_bars)

        x = project_slack(x, slack_mask)
        # Optional: renormalize sin/cos channels to unit norm per node

    return x
