import json

import torch

from SE_torch.data_generator import DataGenerator
from SE_torch.learn_prior.gpt_DSE.data.dataset import sys_to_graph, GraphSample
from SE_torch.learn_prior.gpt_DSE.data.scalers import ChannelScaler
from SE_torch.learn_prior.gpt_DSE.utils.schedules import make_schedule
from SE_torch.learn_prior.gpt_DSE.utils.time_embed import FourierTimeEmbedding
from SE_torch.learn_prior.gpt_DSE.models.eps_gnn import EpsGNN
from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System, Branch

torch.set_default_dtype(torch.float32)

@torch.no_grad()
def _to_phys(x_norm, scaler):
    return scaler.inverse(x_norm)

@torch.no_grad()
def _pack_TV(x_phys, slack_bus_idx):
    V = x_phys[:,0]
    sin_t = x_phys[:,1]
    cos_t = x_phys[:,2]
    T = torch.atan2(sin_t, cos_t).clone()
    if slack_bus_idx is not None: T[slack_bus_idx] = 0.0
    return T.to(torch.float64), V.to(torch.float64)

@torch.no_grad()
def _project_slack_channels(x_norm, slack_mask):
    return x_norm * slack_mask + (1 - slack_mask) * x_norm.detach()

@torch.no_grad()
def meas_grad_from_h(x_norm, scaler, h_ac, z, v, norm_H, slack_bus_idx):
    device = x_norm.device

    x_phys = _to_phys(x_norm, scaler)
    T, V = _pack_TV(x_phys, slack_bus_idx)
    N = x_phys.shape[0]
    R = torch.diag(1.0 / v)

    z_est = h_ac.estimate(T, V)
    delta_z = (z - z_est) / norm_H
    J = h_ac.jacobian(T, V)
    J = torch.cat([J[:, :slack_bus_idx], J[:, slack_bus_idx + 1:]], dim=1)

    JT_R = J.T @ R
    lhs = JT_R @ J
    rhs = JT_R @ delta_z

    delta_x_reduced = torch.linalg.solve(lhs, rhs)

    delta_x = torch.cat([
        delta_x_reduced[:slack_bus_idx],
        torch.tensor([0.0], device=device),
        delta_x_reduced[slack_bus_idx:]
    ])

    delta_T = delta_x[:N]
    delta_V = delta_x[N:]

    sin_T = x_phys[:,1]
    cos_T = x_phys[:,2]
    denom = (sin_T**2 + cos_T**2).clamp_min(1e-8)

    delta_sin_T = delta_T * (cos_T / denom)
    delta_cos_T = delta_T * (-sin_T / denom)
    g_phys = torch.stack([delta_V, delta_sin_T, delta_cos_T], dim=-1)
    std = getattr(scaler, "std", torch.ones(3, device=device)).to(device)
    g_norm = g_phys * std
    return g_norm

@torch.no_grad()
def sample_posterior_guided(graph_template, ckpt_path, cfg, scaler, h_ac, z, v, norm_H,
                            slack_bus_idx, n_steps=50, gamma_max=1.0, ode=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tdiff = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = make_schedule(Tdiff, cfg["diffusion"]["schedule"])
    betas, alphas, alpha_bars = betas.to(device), alphas.to(device), alpha_bars.to(device)
    model = EpsGNN(**cfg["model"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    time_embed = FourierTimeEmbedding(cfg["diffusion"]["time_embed_dim"]).to(device)
    node_meta = graph_template.node_meta.to(device)
    edge_index = graph_template.edge_index.to(device)
    edge_feats = graph_template.edge_feats.to(device)
    slack_mask = graph_template.slack_mask.to(device)
    batch_index = graph_template.batch_index.to(device)

    x_t = graph_template.node_feats.to(device)
    x_t = torch.randn_like(x_t).to(device)
    z = z.to(device).flatten()
    v = v.to(device).flatten()
    norm_H = norm_H.to(device).flatten()

    def gamma_t(step_idx, total):
        import math; s = step_idx / max(1, total - 1)
        return gamma_max * 0.5 * (1 - math.cos(math.pi * s))

    ts = torch.linspace(Tdiff-1, 0, steps=n_steps, dtype=torch.long, device=device)

    for i in range(len(ts)-1):
        x_t = x_t.to(torch.get_default_dtype())
        t = ts[i]
        t_prev = ts[i+1]

        t_nodes = t.expand_as(batch_index)
        t_emb = time_embed(t_nodes, Tdiff)

        ab_t = alpha_bars[t]
        ab_t_prev = alpha_bars[t_prev]
        a_t = alphas[t]
        b_t = betas[t]
        scale = gamma_t(i, len(ts)-1)

        eps_hat = model(x_t, node_meta, edge_index, edge_feats, t_emb, batch_index)
        sigma_t = torch.sqrt(1.0 - ab_t).unsqueeze(-1)
        delta_x_h = meas_grad_from_h(x_t, scaler, h_ac, z, v, norm_H, slack_bus_idx)
        eps_guided = eps_hat #+ scale * delta_x_h

        mean_x_tm1 = (1 / torch.sqrt(a_t)) * (x_t - ((b_t / sigma_t) * eps_guided))
        if t > 0:
            std = torch.sqrt((1 - ab_t_prev) * b_t / sigma_t)
            noise = torch.randn_like(x_t)
            x_tm1 = mean_x_tm1 #+ std * noise
        else:
            x_tm1 = mean_x_tm1
        x_t = _project_slack_channels(x_tm1, slack_mask)
        # sincos = x_t[:,1:3]
        # norm = torch.linalg.norm(sincos, dim=-1, keepdim=True).clamp_min(1e-8)
        # x_t[:,1:3] = sincos / norm
        # x_t = x_tm1
        # x_t[slack_bus_idx, 1:] =
    return x_t
# def sample_posterior_guided(graph_template, ckpt_path, cfg, scaler, h_ac, z, v, norm_H,
#                             slack_bus_idx, n_steps=50, gamma_max=1.0, ode=True):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     Tdiff = cfg["diffusion"]["T"]
#     betas, alphas, alpha_bars = make_schedule(Tdiff, cfg["diffusion"]["schedule"])
#     betas, alphas, alpha_bars = betas.to(device), alphas.to(device), alpha_bars.to(device)
#     model = EpsGNN(**cfg["model"]).to(device)
#     model.load_state_dict(torch.load(ckpt_path, map_location=device))
#     model.eval()
#     time_embed = FourierTimeEmbedding(cfg["diffusion"]["time_embed_dim"]).to(device)
#     node_meta = graph_template.node_meta.to(device)
#     edge_index = graph_template.edge_index.to(device)
#     edge_feats = graph_template.edge_feats.to(device)
#     slack_mask = graph_template.slack_mask.to(device)
#     batch_index = graph_template.batch_index.to(device)
#     x = graph_template.node_feats.to(device)
#     z = z.to(device).flatten(); v = v.to(device).flatten(); norm_H = norm_H.to(device).flatten()
#     def gamma_t(step_idx, total):
#         import math; s = step_idx / max(1, total - 1)
#         return gamma_max * 0.5 * (1 - math.cos(math.pi * s))
#     ts = torch.linspace(Tdiff-1, 0, steps=n_steps, dtype=torch.long, device=device)
#     for i in range(len(ts)-1):
#         x = x.to(torch.get_default_dtype())
#         t = ts[i]; t_prev = ts[i+1]
#         t_nodes = t.expand_as(batch_index)
#         t_emb = time_embed(t_nodes, Tdiff)
#         ab_t = alpha_bars[t]; ab_s = alpha_bars[t_prev]; a_t = alphas[t]; b_t = betas[t]
#         eps_hat = model(x, node_meta, edge_index, edge_feats, t_emb, batch_index)
#         sigma_t = torch.sqrt(1.0 - ab_t).unsqueeze(-1)
#         pred_x0 = (x - sigma_t * eps_hat) / torch.sqrt(ab_t).unsqueeze(-1)
#         g_meas = meas_grad_from_h(pred_x0, scaler, h_ac, z, v, norm_H, slack_bus_idx)
#         g_scale = gamma_t(i, len(ts)-1)
#         eps_guided = eps_hat - g_scale * g_meas
#         mean = 1 / torch.sqrt(a_t) * (x - (b_t / torch.sqrt(1 - ab_t)).unsqueeze(-1) * eps_guided)
#         if t > 0:
#             noise = torch.randn_like(x)
#             x = mean + torch.sqrt(b_t) * noise
#         else:
#             x = mean
#         # pred_x0 = pred_x0 + g_scale * g_meas
#         # x = torch.sqrt(ab_s).unsqueeze(-1) * pred_x0 + torch.sqrt(1.0 - ab_s).unsqueeze(-1) * eps_hat
#         x = _project_slack_channels(x, slack_mask)
#         sincos = x[:,1:3]; norm = torch.linalg.norm(sincos, dim=-1, keepdim=True).clamp_min(1e-8)
#         x[:,1:3] = sincos / norm
#     return x


def main(config_path):
    cfg = json.load(open(config_path))
    net_path = cfg['data']['net_path']
    data = parse_ieee_mat(net_path)
    system_data = data['data']['system']
    sys = System(system_data)
    branch = Branch(sys.branch)
    data_generator = DataGenerator()
    kwargs = {'flow': True, 'injection': True, 'voltage': True, 'current': False,
              'noise': True, 'Pf_noise': 4e-4, 'Qf_noise': 4e-4, 'Cm_noise': 1e-4,
              'Pi_noise': 16e-4, 'Qi_noise': 16e-4, 'Vm_noise': 1.6e-05, 'verbose': True, 'sample': 0.7}
    gen_data = data_generator.generate_measurements(sys, branch, device=None, **kwargs)
    z, v, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true = gen_data
    node_feats, node_meta, edge_index, edge_feats, slack_mask, batch_index = sys_to_graph(sys)
    graph_template = GraphSample(
                    node_feats=node_feats.clone(),
                    node_meta=node_meta.clone(),
                    edge_index=edge_index.clone(),
                    edge_feats=edge_feats.clone(),
                    slack_mask=slack_mask.clone(),
                    batch_index=batch_index.clone()
                )
    ckpt_path = "../training/checkpoints/eps_gnn_uncond_model.pt"
    mean, std = torch.load('../../datasets/mean.pt'), torch.load('../../datasets/std.pt')
    scaler = ChannelScaler(mean=torch.zeros(3), std=torch.ones(3))
    norm_H = torch.ones_like(z)
    slack_bus_idx = sys.slk_bus[0]
    x_est = sample_posterior_guided(
        graph_template, ckpt_path, cfg, scaler, h_ac_polar, z, v, norm_H,
        slack_bus_idx,
    )


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="../configs/default.json")
    args = ap.parse_args()
    main(args.config)