import json
import os
import torch
import torch_geometric as tg
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from SE_torch.learn_prior.DPS.load_data import load_data
from SE_torch.learn_prior.DPS.scheduler import VarScheduler


DEVICE = "cpu"
DTYPE = torch.float32

NUM_EPOCHS = 20
NUM_STEPS = 1000

LEARNING_RATE = 1e-5

NUM_DATA_POINTS = 250000

K_HOPS = 2
HIDDEN_LAYERS = 8
HIDDEN_DIM = 0
T_DIM = 10


class GNN_Model(nn.Module):
    def __init__(self, **kwargs):
        super(GNN_Model, self).__init__()
        self.device = kwargs.get('device', DEVICE)
        self.dtype = kwargs.get('dtype', DTYPE)
        self.in_channels = kwargs.get('in_channels', 2)
        self.hidden_dim = kwargs.get('hidden_dim', HIDDEN_DIM)
        self.out_channels = kwargs.get('out_channels', 2)
        self.k_hops = kwargs.get('k_hops', 2)
        self.hidden_layers = kwargs.get('hidden_layers', HIDDEN_LAYERS)
        self.t_dim = kwargs.get('t_dim', T_DIM)
        self.t_emb = nn.Embedding(NUM_STEPS, self.t_dim).to(device=self.device, dtype=self.dtype)

        p = list(range(self.k_hops))
        self.mix_hops_layers = nn.ModuleList([
            tg.nn.MixHopConv(in_channels=self.in_channels + self.t_dim,
                             out_channels=self.hidden_dim,
                             powers=p).to(device=self.device, dtype=self.dtype)
        ])
        for i in range(self.hidden_layers - 1):
            self.mix_hops_layers.append(
                tg.nn.MixHopConv(in_channels=self.hidden_dim,
                                 out_channels=self.hidden_dim,
                                 powers=p)).to(device=self.device, dtype=self.dtype)
        self.mix_hops_layers.append(
            tg.nn.MixHopConv(in_channels=self.hidden_dim,
                             out_channels=self.out_channels,
                             powers=p).to(device=self.device, dtype=self.dtype)
        )

    def forward(self, x_t, edge_index, t):
        t_embed = self.t_emb(t)
        t_embed = t_embed.expand(x_t.size(0) // t_embed.size(0), -1, -1).transpose(0, 1).reshape(-1, self.t_dim)
        eps = torch.cat([x_t, t_embed], dim=-1)
        for i in range(self.hidden_layers):
            eps = self.mix_hops_layers[i](eps, edge_index)
            eps = eps.view(-1, self.k_hops, self.hidden_dim).sum(dim=1)
            eps = F.relu(eps)
        eps = self.mix_hops_layers[-1](eps, edge_index)
        eps = eps.view(-1, self.k_hops, self.out_channels).sum(dim=1)
        eps = F.relu(eps)
        return eps


def plot_val_loss(train_losses, val_losses):
    epochs = list(range(1, len(val_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='s', label=f'Train Loss - final {train_losses[-1]:.3f}')
    plt.plot(epochs, val_losses, marker='o', label=f'Validation Loss - final {val_losses[-1]:.3f}')
    plt.title("Train and Test Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('./plots/GNU_GNN_train_validation_loss.png')
    plt.close()


class DPS_Trainer:

    def __init__(self, model, data_loaders, **kwargs):

        self.model = model
        self.train_loader, self.test_loader, self.train_dataset = data_loaders

        self.num_steps = kwargs.get('num_steps', NUM_STEPS)
        self.num_epochs = kwargs.get('num_epochs', NUM_EPOCHS)
        self.learning_rate = kwargs.get('learning_rate', LEARNING_RATE)
        self.weight_decay = kwargs.get('weight_decay', 0)

        self.ckpt_path = kwargs.get('ckpt_path')
        self.device = kwargs.get('device', DEVICE)
        self.dtype = kwargs.get('dtype', DTYPE)

        self.var_scheduler = VarScheduler(num_steps=self.num_steps, device=self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = nn.MSELoss().to(device=self.device, dtype=self.dtype)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)


    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        desc = f'{self.model.__class__.__name__} [Epoch {epoch}] 🟢 Training'

        for batch in tqdm(self.train_loader, desc=desc, colour='green'):
            self.optimizer.zero_grad()
            v0, edge_index = batch.x, batch.edge_index
            v0 = v0.view(batch.batch_size, -1, 2).transpose(1, 2).reshape(batch.batch_size, -1)
            v0 = v0.to(device=self.device, dtype=self.dtype)

            edge_index = edge_index.to(device=self.device)

            t = torch.randint(0, self.num_steps, (batch.batch_size,), device=self.device)
            _, _, ab_t = self.var_scheduler[t]
            ab_t = ab_t.unsqueeze(1).to(device=self.device, dtype=self.dtype)

            eps = torch.randn_like(v0)
            x_t = torch.sqrt(ab_t) * v0 + torch.sqrt(1 - ab_t) * eps
            x_t = x_t.view(batch.batch_size, 2, -1).transpose(1, 2).reshape(-1, 2)

            eps_est = self.model(x_t, edge_index, t)
            eps_est = eps_est.view(batch.batch_size, -1, 2).transpose(1, 2).reshape(batch.batch_size, -1)

            loss = self.criterion(eps, eps_est)
            loss.backward()
            self.optimizer.step()
            running_loss += float(loss.item())

        self.scheduler.step()

        return running_loss / max(1, len(self.train_loader))

    def validate_epoch(self, epoch):
        self.model.eval()
        agg_val_loss = 0.0
        desc = f'{self.model.__class__.__name__} [Epoch {epoch}] 🔵 Validating'

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=desc, colour='blue'):
                v0, edge_index = batch.x, batch.edge_index
                v0 = v0.view(batch.batch_size, -1, 2).transpose(1, 2).reshape(batch.batch_size, -1)
                v0 = v0.to(device=self.device, dtype=self.dtype)

                edge_index = edge_index.to(device=self.device)

                t = torch.randint(0, self.num_steps, (batch.batch_size,), device=self.device)
                _, _, ab_t = self.var_scheduler[t]
                ab_t = ab_t.unsqueeze(1).to(device=self.device, dtype=self.dtype)

                eps = torch.randn_like(v0)
                x_t = torch.sqrt(ab_t) * v0 + torch.sqrt(1 - ab_t) * eps
                x_t = x_t.view(batch.batch_size, 2, -1).transpose(1, 2).reshape(-1, 2)

                eps_est = self.model(x_t, edge_index, t)
                eps_est = eps_est.view(batch.batch_size, -1, 2).transpose(1, 2).reshape(batch.batch_size, -1)

                loss = self.criterion(eps, eps_est)
                agg_val_loss += float(loss.item())

        return agg_val_loss / max(1, len(self.test_loader))

    def train(self):
        train_losses, val_losses = [], []

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch {epoch} - Train Loss: {train_loss:.3e}  |  Val Loss: {val_loss:.3e}')

        plot_val_loss(train_losses, val_losses)

        if self.ckpt_path:
            os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.ckpt_path)


def train_DPS_model(config_path):
    config = json.load(open(config_path))
    num_samples = config.get('num_samples', NUM_DATA_POINTS)
    device = config.get('device', DEVICE)
    dtype = config.get('dtype', DTYPE)

    train_loader, test_loader, train_dataset, test_dataset, slk_bus = load_data(config, num_samples)

    model = GNN_Model(**config).to(device=device, dtype=dtype)

    ckpt_path = f"./models/{config.get('ckpt_name')}"
    config['ckpt_path'] = ckpt_path
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    trainer = DPS_Trainer(model, (train_loader, test_loader, train_dataset), **config)
    trainer.train()


if __name__ == "__main__":
    config_pth = f'../configs/DPS_config.json'
    train_DPS_model(config_pth)