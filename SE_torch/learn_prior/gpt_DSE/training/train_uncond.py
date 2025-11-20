
import os, json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from SE_torch.learn_prior.gpt_DSE.data.dataset import GraphDataset, collate_fn
from SE_torch.learn_prior.gpt_DSE.models.eps_gnn import EpsGNN
from SE_torch.learn_prior.gpt_DSE.utils.schedules import make_schedule
from SE_torch.learn_prior.gpt_DSE.utils.time_embed import FourierTimeEmbedding
from SE_torch.learn_prior.gpt_DSE.training.ema import EMA
from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System

torch.set_default_dtype(torch.float32)

def expand_to_nodes(t_graph, batch_index):
    # Map graph-level timesteps to nodes
    return t_graph[batch_index]

def project_slack(x, slack_mask):
    # zero out channels where mask=0 (e.g., sin/cos at slack). Keep others unchanged.
    return x * slack_mask + (1 - slack_mask) * x.detach()

def train(config_path):
    cfg = json.load(open(config_path))
    net_path = cfg['data']['net_path']
    data = parse_ieee_mat(net_path)
    system_data = data['data']['system']
    sys = System(system_data)
    device = "mps" if torch.mps.is_available() else "cpu"
    # device = "cpu"

    # Data
    train_ds = GraphDataset(sys,  n_samples=cfg["data"]["num_samples"])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, collate_fn=collate_fn)

    # Diffusion schedule
    T = cfg["diffusion"]["T"]
    betas, alphas, alpha_bars = make_schedule(T, cfg["diffusion"]["schedule"])
    betas, alphas, alpha_bars = betas.to(device), alphas.to(device), alpha_bars.to(device)

    # Model
    model = EpsGNN(**cfg["model"]).to(device)
    model.load_state_dict(torch.load("checkpoints/eps_gnn_uncond_model.pt"))
    ema = EMA(model, decay=cfg["train"]["ema_decay"])
    criterion = torch.nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(),
                              lr=cfg["train"]["lr"],
                              weight_decay=cfg["train"]["weight_decay"])
    time_embed = FourierTimeEmbedding(cfg["diffusion"]["time_embed_dim"]).to(device)


    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] 🟢 Training", colour='green')
        for batch in pbar:
            # move tensors to device
            x0 = batch["node_feats"].to(device)
            node_meta = batch["node_meta"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_feats = batch["edge_feats"].to(device)
            slack_mask = batch["slack_mask"].to(device)
            batch_index = batch["batch_index"].to(device)

            B = batch_index.max().item() + 1
            t_g = torch.randint(low=0, high=T, size=(B,), device=device)
            t = expand_to_nodes(t_g, batch_index)       # [N_nodes]
            ab = alpha_bars[t]                           # [N_nodes]
            sigma = torch.sqrt(1.0 - ab)

            eps = torch.randn_like(x0)
            xt = torch.sqrt(ab).unsqueeze(-1)*x0 + sigma.unsqueeze(-1)*eps
            t_emb = time_embed(t, T)

            eps_hat = model(xt, node_meta, edge_index, edge_feats, t_emb, batch_index)

            mask = slack_mask
            loss = criterion(eps * mask, eps_hat * mask)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optim.step()
            ema.update(model)

            if step % cfg["train"]["log_interval"] == 0:
                pbar.set_postfix(loss=loss.item())

            step += 1
            train_loss += loss
        print(f'Epoch {epoch} - Train Loss: {train_loss:.3f}')

    # Save EMA weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(ema.unwrap().state_dict(), "checkpoints/eps_gnn_uncond_ema.pt")
    torch.save(model.state_dict(), "checkpoints/eps_gnn_uncond_model.pt")
    # print("Saved EMA checkpoint to checkpoints/eps_gnn_uncond_ema.pt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="../configs/default.json")
    args = ap.parse_args()
    train(args.config)
