import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence

from tqdm.auto import tqdm

from SE_torch.learn_prior.NF.load_data import load_data

DEVICE = torch.device('mps' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

NUM_EPOCHS = 30
LR = 1e-3
K_HOPS = 2
HIDDEN_LAYERS = 8
IN_CHANNELS = 2
OUT_CHANNELS = 2
IN_FEATURES = 236
LATENT_SAMPLES = 10
LATENT_DIM = 200
NUM_DATA_POINTS = 25000


class VAESimpleEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(VAESimpleEncoder, self).__init__()
        self.device = kwargs.get("device", DEVICE)
        self.dtype = kwargs.get("dtype", torch.get_default_dtype())

        self.in_features = kwargs.get("in_features", IN_FEATURES)
        self.latent_dim = kwargs.get("latent_dim", LATENT_DIM)

        hidden_dims = kwargs.get("hidden_dims", [512, 512, 256])  # you can change
        act = kwargs.get("activation", "leaky_relu")  # "relu" / "tanh"
        dropout = kwargs.get("dropout", 0.0)

        self.act_name = act
        self.dropout = float(dropout)

        layers = []
        d = self.in_features
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.LeakyReLU(1e-2) if act == "leaky_relu" else nn.ReLU() if act == "relu" else nn.Tanh())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            d = h
        self.backbone = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(d, self.latent_dim)
        self.fc_logvar = nn.Linear(d, self.latent_dim)

        if self.device is not None:
            self.to(self.device, dtype=self.dtype)

    def forward(self, x, edge_index=None):
        h = self.backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, -20.0, 10.0)
        return mu, logvar


class VAESimpleDecoder(nn.Module):
    def __init__(self, **kwargs):
        super(VAESimpleDecoder, self).__init__()
        self.device = kwargs.get("device", DEVICE)
        self.dtype = kwargs.get("dtype", torch.get_default_dtype())

        self.in_features = kwargs.get("in_features", IN_FEATURES)
        self.latent_dim = kwargs.get("latent_dim", LATENT_DIM)

        hidden_dims = kwargs.get("hidden_dims", [512, 512, 256]) # usually mirror encoder
        hidden_dims.reverse()
        act = kwargs.get("activation", "leaky_relu")
        dropout = kwargs.get("dropout", 0.0)

        self.act_name = act
        self.dropout = float(dropout)

        layers = []
        d = self.latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            # layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(1e-2) if act == "leaky_relu" else nn.ReLU() if act == "relu" else nn.Tanh())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            d = h

        layers.append(nn.Linear(d, self.in_features))
        self.net = nn.Sequential(*layers)

        if self.device is not None:
            self.to(self.device, dtype=self.dtype)

    def forward(self, z):
        x_hat = self.net(z)
        return x_hat


class AmortizedVAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get('device', DEVICE)
        self.latent_dim = kwargs.get('latent_dim', LATENT_DIM)
        self.decoder = VAESimpleDecoder(**kwargs).to(self.device)
        self.encoder = VAESimpleEncoder(**kwargs).to(self.device)

    def encode(self, x, **kwargs):
        return self.encoder(x, **kwargs)

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(logvar * .5 )
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def decode(self, z, **kwargs):
        return self.decoder(z, **kwargs)

    def forward(self, x, **kwargs):
        mu, logvar = self.encode(x, **kwargs)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, **kwargs), mu, logvar


def KLLoss(mu, logvar):
    P = Normal(mu, torch.exp(logvar / 2))
    Q = Normal(torch.zeros_like(mu), 1)
    kl_loss = torch.mean(kl_divergence(P, Q))
    return kl_loss


def compute_loss(model_output, target, MSE_loss):
    recon_x, mu, logvar = model_output
    loss = MSE_loss(recon_x, target) + KLLoss(mu, logvar)
    return loss


class TrainingConfig:
    def __init__(self, **kwargs):
        self.num_epochs = kwargs.get('num_epochs', NUM_EPOCHS)
        self.validate = kwargs.get('validate', False)
        self.lr = kwargs.get('learning_rate', LR)
        self.latent_samples = kwargs.get('latent_samples', LATENT_SAMPLES)
        self.ckpt_name = kwargs.get('ckpt_name')
        self.device = kwargs.get('device', DEVICE)
        self.dtype = kwargs.get('dtype', torch.get_default_dtype())


class VAETrainer:
    def __init__(self, model, config, data_loaders):
        self.model = model
        self.config = config
        self.train_loader, self.test_loader, self.train_dataset, self.test_dataset, _ = data_loaders
        self.criterion = nn.MSELoss()
        self.prior_dist = torch.randn(config.latent_samples, model.latent_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        desc = f'{self.model.__class__.__name__} [Epoch {epoch}] 🟢 Training'

        for batch in tqdm(self.train_loader, desc=desc, colour='green'):
            self.optimizer.zero_grad()
            x = batch[0].to(self.config.device, dtype=self.config.dtype)
            x.requires_grad = True
            model_output = self.model(x)
            loss = compute_loss(model_output, x, self.criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        desc = f'{self.model.__class__.__name__} [Epoch {epoch}] 🔵 Validating'

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=desc, colour='blue'):
                x = batch[0].to(self.config.device, dtype=self.config.dtype)
                model_output = self.model(x)
                loss = compute_loss(model_output, x, self.criterion)
                val_loss += loss.item()

        return val_loss / len(self.test_loader)

    def train(self):
        train_losses = []
        val_losses = []

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            # self.scheduler.step()
            if self.config.validate:
                val_loss = self.validate_epoch(epoch)
                val_losses.append(val_loss)
                print(f'Epoch {epoch} - Train Loss: {train_loss:.7f} - Val Loss: {val_loss:.7f}')
            else:
                print(f'Epoch {epoch} - Train Loss: {train_loss:.7f}')

        self.save_results()

    def save_results(self):
        os.makedirs('./models', exist_ok=True)
        ckpt_path = f"./models/{self.config.ckpt_name}"
        torch.save(self.model.state_dict(), ckpt_path)


def compute_log_prob(x, model, num_samples=1000, sigma_p=0.4):
    mu, logvar = model.encode(torch.unsqueeze(x, 0))

    prior_dist = Independent(Normal(torch.zeros((1, model.latent_dim)),
                             torch.ones((1, model.latent_dim))), 2)
    posterior_dist = Independent(Normal(mu, torch.exp(logvar / 2)), 2)

    z_samples = posterior_dist.sample([num_samples])
    log_prior = prior_dist.log_prob(z_samples)
    log_posterior = posterior_dist.log_prob(z_samples)

    decoded = model.decode(z_samples)
    log_reconstruction = Independent(Normal(decoded, sigma_p), 3).log_prob(x)

    log_prob = torch.logsumexp(log_prior + log_reconstruction - log_posterior, 0) - np.log(num_samples)

    return log_prob


def main(config_path):
    config = json.load(open(config_path))
    num_samples = config.get('num_samples', NUM_DATA_POINTS)
    device = config.get('device', DEVICE)
    dtype = config.get('dtype', DTYPE)
    train_config = TrainingConfig(**config)
    data = load_data(config, num_samples)
    model = AmortizedVAE(**config).to(device, dtype)

    ckpt_path = f"./models/{config.get('ckpt_name')}"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    trainer = VAETrainer(model, train_config, data)
    trainer.train()


if __name__ == '__main__':
    for config_name in os.listdir("../configs"):
        if config_name.startswith("VAE_v0.5"):
            print(config_name)
            config_pth = f'../configs/{config_name}'
            main(config_pth)
