import torch
from torch import nn
import torch.nn.functional as F
from hypmath import wrapped_normal, poincareball


class VariationalEncoder(nn.Module):
    """
    Hyperbolic encoder architecture for the MNIST data
    """

    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)
        self.kl = 0

    def forward(self, x):
        # x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = poincareball.PoincareBall(self.latent_dims).expmap0(x)
        x = self.linear1(x)
        x = poincareball.PoincareBall(self.latent_dims).logmap0(x)
        mu = poincareball.PoincareBall(
            self.latent_dims).expmap0(self.linear2(x))
        sigma = F.softplus(torch.exp(self.linear3(x)))
        self.N = wrapped_normal.WrappedNormal(
            mu, sigma, poincareball.PoincareBall(self.latent_dims))
        # reparametrisation trick
        z = self.N.rsample()
        # K-L divergence
        self.kl = ((sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum())
        return z


class Decoder(nn.Module):
    """
    Hyperbolic decoder architecture for the MNIST data
    """

    def __init__(self, latent_dims):
        super().__init__()

        self.latent_dims = latent_dims
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = poincareball.PoincareBall(self.latent_dims).logmap0(x)
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class VariationalAutoencoder(nn.Module):
    """
    Full hyperbolic VAE architecture incorporating encoder and decoder
    Parameter:
    latent_dims: dimension of latent space
    """

    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
