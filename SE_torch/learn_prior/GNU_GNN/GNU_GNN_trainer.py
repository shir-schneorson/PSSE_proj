import json
import os
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from SE_torch.learn_prior.GNU_GNN.models2 import GNU_Model
from SE_torch.learn_prior.GNU_GNN.load_data import load_data

DEVICE = "mps"
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

NUM_EPOCHS = 30
NUM_STEPS = 10

LEARNING_RATE = 1e-3
PSI_STEP_SIZE = 1e-2

NUM_DATA_POINTS = 250000

GAMMA = 1
RHO = 1


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


class Psi(nn.Module):
    def __init__(self, step_size=PSI_STEP_SIZE, gamma=GAMMA, rho=RHO):
        super(Psi, self).__init__()
        self.step_size = step_size
        self.gamma = gamma
        self.rho = rho
        self.c = nn.MSELoss()
        self.loss = nn.MSELoss()

    def forward(self, model, z, v_true, edge_index):
        zeta = z.clone().detach()
        v = v_true.clone().detach()
        v.requires_grad = False
        zeta.requires_grad = True
        v_est, _ = model(zeta, edge_index)
        psi_loss = self.loss(v_est, v) + self.gamma * (self.rho - self.c(z, zeta))
        psi_loss.backward()
        zeta = zeta + self.step_size * zeta.grad
        return zeta


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, v_est, v_true):
        v_est = v_est.clone().detach().to('cpu')
        v_true = v_true.clone().detach().to('cpu')
        half_dim = v_true.shape[1] // 2
        X_true = v_true.view(-1, 2, half_dim).transpose(1, 2).reshape(-1, 2)
        T_true, V_true = X_true[:, 0], X_true[:, 1]

        X_est = v_est.view(-1, 2, half_dim).transpose(1, 2).reshape(-1, 2)
        T_est, V_est = X_est[:, 0], X_est[:, 1]

        u_true = (V_true * torch.exp(1j * T_true)).reshape(-1, half_dim)
        u_est = (V_est * torch.exp(1j * T_est)).reshape(-1, half_dim)

        res = u_est - u_true
        num = torch.norm(res, dim=1)
        den = torch.norm(u_true, dim=1)

        err = (num / den)
        return err

class GNU_GNN_Trainer:

    def __init__(self, GNU_model, data_loaders,
                 num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
                 ckpt_path='', device='cpu', dtype=torch.get_default_dtype()):
        self.GNU_model = GNU_model
        self.train_loader, self.test_loader, self.train_dataset = data_loaders

        self.optimizer = optim.Adam(self.GNU_model.parameters(), lr=learning_rate)
        self.psi = Psi().to(device=device, dtype=dtype)
        self.criterion = torch.nn.MSELoss(reduction='sum').to(device=device, dtype=dtype)
        self.rmse = RMSE().to(device='cpu', dtype=dtype)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        self.num_epochs = num_epochs
        self.ckpt_path = ckpt_path
        self.device = device
        self.dtype = dtype

    def train_epoch(self, epoch):
        self.GNU_model.train()
        running_loss = 0.0
        desc = f'{self.GNU_model.__class__.__name__} [Epoch {epoch}] 🟢 Training'
        pbar = tqdm(self.train_loader, desc=desc, colour='green')
        for batch in pbar:
            self.optimizer.zero_grad()
            v_true, edge_index, z = batch.x, batch.edge_index, batch.y
            z = z.view(batch.batch_size, -1, 2)[:, :, 0].squeeze(-1)
            z = z.to(device=self.device, dtype=self.dtype)
            v_true = v_true.to(device=self.device, dtype=self.dtype)
            v_true = v_true.view(batch.batch_size, -1, 2).transpose(1, 2).reshape(batch.batch_size, -1)
            edge_index = edge_index.to(device=self.device)

            zeta = self.psi(self.GNU_model, z, v_true, edge_index)
            v_est, _ = self.GNU_model(zeta, edge_index)

            loss = (v_est - v_true).pow(2).sum(dim=1).mean()
            loss.backward()
            self.optimizer.step()
            running_loss += float(loss.item())
            with torch.no_grad():
                rmse = self.rmse(v_est, v_true)
                pbar.set_postfix(loss=loss.item(), rmse_min=rmse.min().item(), rmse_max=rmse.max().item())

        self.scheduler.step()

        return running_loss / max(1, len(self.train_loader))

    def validate_epoch(self, epoch):
        self.GNU_model.eval()
        agg_val_loss = 0.0
        desc = f'{self.GNU_model.__class__.__name__} [Epoch {epoch}] 🔵 Validating'

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=desc, colour='blue')
            for batch in pbar:
                v_true, edge_index, z = batch.x, batch.edge_index, batch.y
                z = z.view(batch.batch_size, -1, 2)[:, :, 0].squeeze(-1)
                z = z.to(device=self.device, dtype=self.dtype)
                v_true = v_true.to(device=self.device, dtype=self.dtype)
                v_true = v_true.view(batch.batch_size, -1, 2).transpose(1, 2).reshape(batch.batch_size, -1)
                edge_index = edge_index.to(device=self.device)

                v_est, all_v = self.GNU_model(z, edge_index)

                loss = (v_est - v_true).pow(2).sum(dim=1).mean()
                agg_val_loss += float(loss.item())
                rmse = self.rmse(v_est, v_true)
                pbar.set_postfix(loss=loss.item(), rmse_min=rmse.min().item(), rmse_max=rmse.max().item())

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
            torch.save(self.GNU_model.state_dict(), self.ckpt_path)


def train_GNU_GNN_model(config_path):
    config = json.load(open(config_path))
    num_samples = config.get('num_samples', NUM_DATA_POINTS)
    device = config.get('device', DEVICE)
    dtype = config.get('dtype', DTYPE)

    train_loader, test_loader, train_dataset, test_dataset, config = load_data(config, num_samples)

    model = GNU_Model(**config).to(device=device, dtype=dtype)

    ckpt_path = f"./models/{config.get('ckpt_name')}"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    trainer = GNU_GNN_Trainer(model, (train_loader, test_loader, train_dataset),
                              ckpt_path=ckpt_path, device=device, dtype=dtype)
    trainer.train()


if __name__ == "__main__":
    for config_name in os.listdir("../configs"):
        if config_name.startswith("GNU_GNN_obs"):
            print(config_name)
            config_pth = f'../configs/{config_name}'
            train_GNU_GNN_model(config_pth)