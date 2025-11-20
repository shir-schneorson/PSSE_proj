import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import dtype
from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from SE_torch.learn_prior.GNU_GNN.models import GNU_Model
from SE_torch.learn_prior.GNU_GNN.load_data import load_data

# DEVICE = (
#     torch.device("mps") if torch.backends.mps.is_available()
#     else torch.device("cuda") if torch.cuda.is_available()
#     else torch.device("cpu")
# )

DEVICE = "cpu"
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

NUM_EPOCHS = 20
NUM_STEPS = 10

LEARNING_RATE = 1e-5
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

    def forward(self, model, z, v_true):
        zeta = z.clone().requires_grad_()
        psi_loss = self.loss(model(zeta), v_true) + self.gamma * (self.rho - self.c(z, zeta))
        grad_psi_zeta = torch.autograd.grad(psi_loss, zeta)[0]
        return z + self.step_size * grad_psi_zeta


class GNU_GNN_Trainer:

    def __init__(self, GNU_model, data_loaders,
                 num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
                 ckpt_path='', device='cpu', dtype=torch.get_default_dtype()):
        self.GNU_model = GNU_model
        self.train_loader, self.test_loader, self.train_dataset, self.data_mean, self.data_cov, self.data_std = data_loaders

        self.optimizer = optim.Adam(self.GNU_model.parameters(), lr=learning_rate)
        self.psi = Psi()
        self.criterion = nn.MSELoss()

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        self.simple_data_dist = MultivariateNormal(self.data_mean, self.data_cov)
        self.num_epochs = num_epochs
        self.ckpt_path = ckpt_path
        self.device = device
        self.dtype = dtype

    def train_epoch(self, epoch):
        self.GNU_model.train()
        running_loss = 0.0
        desc = f'{self.GNU_model.__class__.__name__} [Epoch {epoch}] 🟢 Training'

        for batch in tqdm(self.train_loader, desc=desc, colour='green'):
            self.optimizer.zero_grad()
            z, v_true = batch
            z = z.to(device=self.device, dtype=dtype)
            v_true = v_true.to(device=self.device, dtype=dtype)

            zeta = self.psi(self.GNU_model, z, v_true)
            v_est = self.GNU_model(zeta)

            loss = self.criterion(v_est, v_true)
            loss.backward()
            self.optimizer.step()
            running_loss += float(loss.item())

        self.scheduler.step()

        return running_loss / max(1, len(self.train_loader))

    def validate_epoch(self, epoch):
        self.GNU_model.eval()
        agg_val_loss = 0.0
        desc = f'{self.GNU_model.__class__.__name__} [Epoch {epoch}] 🔵 Validating'

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=desc, colour='blue'):
                z, v_true = batch
                z = z.to(device=self.device, dtype=dtype)

                v_true = v_true.to(device=self.device, dtype=dtype)
                v_est = self.GNU_model(z)

                loss = self.criterion(v_est, v_true)
                agg_val_loss += float(loss.item())

        return agg_val_loss / max(1, len(self.test_loader))

    def train(self):
        train_losses, val_losses = [], []

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch {epoch} - Train Loss: {train_loss:.3f}  |  Val Loss: {val_loss:.3f}')

        plot_val_loss(train_losses, val_losses)

        if self.ckpt_path:
            os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
            torch.save(self.GNU_model.state_dict(), self.ckpt_path)


def train_GNU_GNN_model(config_path):
    config = json.load(open(config_path))
    num_samples = config.get('num_samples', NUM_DATA_POINTS)
    device = config.get('device', DEVICE)
    dtype = config.get('dtype', DTYPE)

    train_loader, test_loader, train_dataset, test_dataset, mean, cov, std = load_data(config, num_samples)

    model = GNU_Model(**config).to(device=device, dtype=dtype)

    ckpt_path = f"./models/{config.get('ckpt_name')}"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    trainer = GNU_GNN_Trainer(model, (train_loader, test_loader, train_dataset, mean, cov, std),
                              ckpt_path=ckpt_path, device=device, dtype=dtype)
    trainer.train()


if __name__ == "__main__":
    for config_name in os.listdir("../configs"):
        config_pth = f'./configs/{config_name}'
        train_GNU_GNN_model(config_pth)