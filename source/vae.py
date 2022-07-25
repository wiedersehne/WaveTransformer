import logging
from abc import ABC
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from experiments.configs.config import extern
from .embedding_module import EmbeddingModule
from .classifier import SequenceModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearEncoder(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.encoder1 = nn.Linear(in_features=in_features, out_features=64)
        self.enc1 = nn.Linear(in_features=64, out_features=264)
        self.enc2 = nn.Linear(in_features=264, out_features=264)
        self.enc3 = nn.Linear(in_features=264, out_features=64)
        self.encoder2 = nn.Linear(in_features=64, out_features=out_features)

    def forward(self, x):
        x = F.relu(self.encoder1(x))
        x = F.relu(self.enc3(F.relu(self.enc2(F.relu(self.enc1(x))))))

        x = self.encoder2(x)
        return x


class LinearDecoder(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.decoder1 = nn.Linear(in_features=in_features, out_features=64)
        self.dec1 = nn.Linear(in_features=64, out_features=264)
        self.dec2 = nn.Linear(in_features=264, out_features=264)
        self.dec3 = nn.Linear(in_features=264, out_features=64)
        self.decoder2 = nn.Linear(in_features=64, out_features=out_features)

    def forward(self, x):
        x = F.relu(self.decoder1(x))
        x = F.relu(self.dec3(F.relu(self.dec2(F.relu(self.dec1(x))))))
        channel1 = torch.sigmoid(self.decoder2(x))
        return channel1


class CoefficientDecoder(nn.Module):
    def __init__(self, in_features, seq_length, kernels):
        super().__init__()
        # unzip all bases from kernels
        bases = np.stack([[base[1] for base in class_kernels] for class_kernels in kernels])
        self.bases = torch.Tensor(bases.reshape((-1, seq_length))).to(device)

        self.decoder1 = nn.Linear(in_features=in_features, out_features=64)
        self.dec1 = nn.Linear(in_features=64, out_features=64)
        self.dec2 = nn.Linear(in_features=64, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=64)
        self.decoder2 = nn.Linear(in_features=64, out_features=self.bases.shape[0])

    def forward(self, x):

        x = F.relu(self.decoder1(x))

        x = self.dec1(self.dec2(self.dec3(x)))

        z = self.decoder2(x)
        channel1 = torch.zeros((z.shape[0], self.bases.shape[1])).to(device)
        for idx_base in range(self.bases.shape[0]):
            for idx_n in range(channel1.shape[0]):
                channel1[idx_n, :] += z[idx_n, idx_base] * self.bases[idx_base, :]


        #channel1 = torch.matmul(channel1, self.bases)

        #channel2 = F.relu(self.decoder2_2(x))
        #channel2 = torch.matmul(channel2, self.bases)

        #reconstruction = torch.stack([channel1, channel1], dim=2)
        #reconstruction = reconstruction.unsqueeze(2)
        return channel1


class VanillaVAE(pl.LightningModule, ABC, EmbeddingModule):

    def __init__(self, seq_length: int,
                 latent_dim: int,
                 kernels=None,
                 enc_hidden: int = 256,
                 n_layers: int = 3,
                 dropout: float = 0,
                 bidirectional: bool = True,
                 stack: int = 1,
                 in_channels: int = 23,
                 out_channels: int = 23,
                 kernel_size: int = 3,
                 stride: int = 5,
                 padding: int = 1,
                 ):
        """

        :param latent_dim:
        :param enc_hidden:
        :param n_layers:
        :param dropout:
        :param bidirectional:
        :param stack:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        """
        self.auto_encoder = True

        self.h_dim = latent_dim
        self.seq_length = seq_length
        self.kernels = kernels

        super().__init__()

        # Encoder - (Bidirectional) LSTM
        self.kw_dict = {"n_hidden": enc_hidden,
                        "n_layers": n_layers,
                        "dropout": dropout,
                        "bidirectional": bidirectional,
                        "stack": stack,
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "kernel_size": kernel_size,
                        "stride": stride,
                        "padding": padding
                        }
        self.encoder = SequenceModel(seq_length=self.seq_length, n_classes=self.h_dim, **self.kw_dict)
        self.fc_mu = nn.Linear(self.h_dim, self.h_dim)
        self.fc_var = nn.Linear(self.h_dim, self.h_dim)

        # Decoder
        if self.kernels is not None:
            print("Using Co-efficient decoder, learning co-efficients of known basis")
            self.decoder = CoefficientDecoder(in_features=self.h_dim, seq_length=self.seq_length, kernels=self.kernels)
        else:
            print("Using linear decoder")
            self.decoder = LinearDecoder(in_features=self.h_dim, out_features=self.seq_length)

        self.criterion = nn.MSELoss()

    def load_checkpoint(self, path):
        arg_dict = {"latent_dim": self.h_dim, "seq_length": self.seq_length, "kernels": self.kernels}
        return self.load_from_checkpoint(path, **arg_dict, **self.kw_dict)

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
        print(recon.shape)
        return [recon, x[:, :, 1, 0], mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        # self.plot_results(args)

        recons, input, mu, log_var = args[0], args[1], args[2], args[3]
        print(input.shape)
        kld_weight = 64  # TODO: kwargs['M_N']  # Account for the minibatch samples from the dataset
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


@extern
def model_constructor(latent_dim, seq_length, kernels=None, bidirectional=True, lstm_dropout=0., dir_path="./experiments/logs",
                      verbose=False, monitor="val_loss", mode="min", num_epochs=100, gpus=1, pb_refresh=10,
                      enc_hidden=256, n_layers=3, stack=1, in_channels=23, out_channels=23, kernel_size=3, stride=5,
                      padding=1):
    """Wrapper. Decorated for .yaml configuration and readability """
    model_ = VanillaVAE(
        latent_dim=latent_dim,
        seq_length=seq_length,
        kernels=kernels,
        enc_hidden=enc_hidden,
        n_layers=n_layers,
        dropout=lstm_dropout,
        bidirectional=bidirectional,
        stack=stack,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

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
