import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import Trainer


class VAE(pl.LightningModule):

    def __init__(self, alpha=1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 196), nn.ReLU(), nn.BatchNorm1d(196, momentum=0.7),
                                     nn.Linear(196, 49), nn.ReLU(), nn.BatchNorm1d(
                                         49, momentum=0.7),
                                     nn.Linear(49, 28), nn.LeakyReLU())
        self.hidden2mu = nn.Linear(28, 28)
        self.hidden2log_var = nn.Linear(28, 28)
        self.alpha = alpha
        self.decoder = nn.Sequential(nn.Linear(28, 49), nn.ReLU(),
                                     nn.Linear(49, 196), nn.ReLU(),
                                     nn.Linear(196, 784), nn.Tanh())
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))])

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1)))
        z = z.type_as(mu)  # Setting z to be .cuda when using GPU training
        return mu + sigma*z

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        return self.decoder(hidden)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        mu, log_var = self.encode(x)

        kl_loss = (-0.5*(1+log_var - mu**2 -
                   torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        hidden = self.reparametrize(mu, log_var)
        x_out = self.decode(hidden)

        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(),recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss

        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        mu, log_var = self.encode(x)

        kl_loss = (-0.5*(1+log_var - mu**2 -
                   torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        hidden = self.reparametrize(mu, log_var)
        x_out = self.decode(hidden)

        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(),recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss
        self.log('val_kl_loss', kl_loss, on_step=True, on_epoch=True)
        self.log('val_recon_loss', recon_loss, on_step=True, on_epoch=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        return x_out, loss

    def scale_image(self, img):
        out = (img + 1) / 2
        return out

    def train_dataloader(self):
        mnist_train = datasets.MNIST(
            'PATH_TO_STORE_TRAINSET', download=True, train=True, transform=self.data_transform)
        return DataLoader(mnist_train, batch_size=64)

    def val_dataloader(self):
        mnist_val = datasets.MNIST(
            'PATH_TO_STORE_TESTSET', download=True, train=False, transform=self.data_transform)
        return DataLoader(mnist_val, batch_size=64)

    def validation_epoch_end(self, outputs):
        if not os.path.exists('vae_images'):
            os.makedirs('vae_images')
        choice = random.choice(outputs)  # Choose a random batch from outputs
        output_sample = choice[0]  # Take the recreated image
        # Reshape tensor to stack the images nicely
        output_sample = output_sample.reshape(-1, 1, 28, 28)
        output_sample = self.scale_image(output_sample)
        save_image(output_sample,
                   f"vae_images/epoch_{self.current_epoch+1}.png")


def main():
    trainer = Trainer(gpus=0, auto_lr_find=True, max_epochs=25)
    trainer.fit(VAE())


if __name__ == "__main__":
    main()
