from abc import ABC
import numpy as np
# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# PyTorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# Local
from experiments.configs.config import extern
from source.models.sequence_encoder import SequenceEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VanillaVAE(pl.LightningModule, ABC):

    def __init__(self,
                 encoder_setup: dict,
                 decoder_model,
                 latent_dim: int,
                 kld_weight=None,
                 ):
        """


        """

        super().__init__()
        self.kld_weight = kld_weight
        self.save_hyperparameters()

        # Encoder
        self.encoder = SequenceEncoder(**encoder_setup)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

        # Decoder
        self.decoder = decoder_model

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(path)

    def encode(self, x: torch.tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder [N x L x C x 2]
        :return: (Tensor) List of latent codes
        """
        hidden = self.encoder(x)
        return [self.fc_mu(hidden), self.fc_var(hidden)]

    def decode(self, z: torch.tensor):
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder(z)

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.tensor, **kwargs):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return [recon, x[:, 0, 0, :], mu, log_var]

    def loss_function(self, *args) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        # self.plot_results(args)

        recons, input, mu, log_var = args[0], args[1], args[2], args[3]
        kld_weight = input.size(0) if self.kld_weight is None else self.kld_weight
        recons_loss = F.mse_loss(input, recons)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    @staticmethod
    def plot_results(results):
        import matplotlib.pyplot as plt
        plt_idx = np.random.randint(0, results[0].shape[0])
        plt.scatter(np.linspace(0, 20, 20), results[1].detach().cpu()[plt_idx, :])
        plt.scatter(np.linspace(0, 20, 20), results[0].detach().cpu()[plt_idx, :].cpu())
        plt.pause(0.1)
        plt.close()

    def training_step(self, batch, batch_idx):
        # print(batch['feature'].shape)    # N x seq_len x seg_length=1 x num_channels=2
        # print(batch['label'].shape)      # N
        sequences, labels = batch['feature'], batch['label']
        results = self(sequences)
        train_loss_dict = self.loss_function(*results)
        self.log("train_loss", train_loss_dict['loss'], prog_bar=True, logger=True)
        self.log("train_recon_loss", train_loss_dict['Reconstruction_Loss'], prog_bar=True, logger=True)
        self.log("train_KLD", train_loss_dict['KLD'], prog_bar=True, logger=True)
        #self.plot_results(results)
        return train_loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        results = self(sequences)
        val_loss_dict = self.loss_function(*results)
        self.log("val_loss", val_loss_dict['loss'], prog_bar=True, logger=True)
        self.log("val_recon_loss", val_loss_dict['Reconstruction_Loss'], prog_bar=True, logger=True)
        self.log("val_KLD", val_loss_dict['KLD'], prog_bar=True, logger=True)
        #self.plot_results(results)
        return val_loss_dict['loss']

    def test_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        results = self(sequences)
        test_loss_dict = self.loss_function(*results)
        self.log("test_loss", test_loss_dict['loss'], prog_bar=True, logger=True)
        self.log("test_recon_loss", test_loss_dict['Reconstruction_Loss'], prog_bar=True, logger=True)
        self.log("test_KLD", test_loss_dict['KLD'], prog_bar=True, logger=True)
        return test_loss_dict['loss']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer),         # The scheduler instance
            "interval": "epoch",                               # The unit of the scheduler's step size
            "frequency": 1,                     # How many epochs/steps should pass between calls to `scheduler.step()`
            "monitor": "val_loss",              # Metric to monitor for scheduler
            "strict": True,                     # Enforce that "val_loss" is available when the scheduler is updated
            "name": 'LearningRateMonitor',      # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }


def create_vanilla_vae(encoder_setup, decoder_model, latent_dim, kld_weight,
                       dir_path="./experiments/logs", verbose=False, monitor="val_loss", mode="min",
                       num_epochs=100, gpus=1, pb_refresh=10,
                       ):
    """Wrapper. Decorated for .yaml configuration and readability """
    model_ = VanillaVAE(encoder_setup=encoder_setup, decoder_model=decoder_model,
                        latent_dim=latent_dim, kld_weight=kld_weight)

    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_path + "/checkpoints",
        filename='best_checkpoint',
        save_top_k=1,
        verbose=verbose,
        monitor=monitor,
        mode=mode,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", mode="min",
        # min_delta=0.00,
        patience=5,
        verbose=verbose
    )

    trainer_ = pl.Trainer(
        default_root_dir=dir_path,
        callbacks=[checkpoint_callback, early_stop_callback],
        checkpoint_callback=True,
        max_epochs=num_epochs,
        check_val_every_n_epoch=1,
        gpus=gpus,
        progress_bar_refresh_rate=pb_refresh,
    )

    return model_, trainer_
