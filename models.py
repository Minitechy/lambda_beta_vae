# models.py
"""Model definitions for the Convolutional VAE."""

import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    """Convolutional encoder for VAE."""
    def __init__(self, latent_dim, input_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        h = torch.relu(self.fc1(h))
        return self.fc_mu(h), self.fc_logvar(h)

class ConvDecoder(nn.Module):
    """Convolutional decoder for VAE."""
    def __init__(self, latent_dim, num_channels, use_sigmoid):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 64 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_channels, 4, stride=2, padding=1)
        )
        self.use_sigmoid = use_sigmoid

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h)).view(z.size(0), 64, 4, 4)
        h = self.decoder(h)
        return torch.sigmoid(h) if self.use_sigmoid else h

class VAE(nn.Module):
    """VAE model combining encoder and decoder."""
    def __init__(self, latent_dim, input_channels, loss_type):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, input_channels)
        self.decoder = ConvDecoder(latent_dim, input_channels, use_sigmoid=(loss_type == 'bce'))
        self.loss_type = loss_type

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=std.device)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar