from abc import ABC
# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# PyTorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger              # tracking tool
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import ptwt
import pywt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(pl.LightningModule, ABC):

    @staticmethod
    def fwt(x):
        # Fast wavelet transform
        x_flat = torch.reshape(x, (x.size(0), -1))
        wavelet = pywt.Wavelet('haar')
        return ptwt.wavedec(x_flat, wavelet, mode='zero', level=2)

    def __init__(self,
                 encoder_model,
                 decoder_model,
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.embed_dim, self.layers, self.hidden_size = 2, 2, 128  # TODO, don't hard code

        # Encoder
        self.encoder = encoder_model
        # Connect embedded latent dimension to the initial hidden state of the decoder
        self.fc1 = nn.Linear(self.embed_dim, self.layers*1 * self.hidden_size)
        self.fc2 = nn.Linear(self.embed_dim, self.layers*1 * self.hidden_size)
        # Decoder
        self.decoder = decoder_model

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(path)

    # def encode(self, x: torch.tensor):
    #     """
    #     Encodes the input by passing through the encoder network
    #     and returns the latent codes.
    #     :param x: (Tensor) Input tensor to encoder
    #     :return: (Tensor) List of latent codes
    #     """
    #     return self.encoder(x)
    #
    # def decode(self, z: torch.tensor):
    #     """
    #     Maps the given latent codes onto the feature space.
    #     :param z: (Tensor) [B x D]
    #     :return: (Tensor) [B x Strands x Chromosomes x L]
    #     """
    #     return self.decoder(z)

    def forward(self, x: torch.tensor, teacher_forcing=0.5, **kwargs):
        """
        x,                 features = [B, Strands, Chromosomes, Sequence Length]
        teacher_forcing,   the teacher forcing ratio for our RNN decoder (used in training)
        """

        batch_size = x.shape[0]
        target_stack = self.fwt(torch.reshape(x, (x.shape[0], -1)))                      # Filter bank
        trg_length = len(target_stack)                  # Length of target sequence
        trg_seq_len = [seq.shape[1] for seq in target_stack]    # Length of each sequence in the bank

        # tensor to store decoder outputs, padding with zeros
        outputs = [torch.zeros(batch_size, trg_seq_len[-1]) for _ in target_stack]

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        z = self.encoder(x)
        hidden = torch.reshape(self.fc1(z), (z.shape[0], self.layers, self.hidden_size)).permute((1, 0, 2))
        cell = torch.reshape(self.fc2(z), (z.shape[0], self.layers, self.hidden_size)).permute((1, 0, 2))
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # print(hidden.shape)
        # print(cell.shape)

        recon = self.decoder(z)
        # print(x.shape)
        # print(recon.shape)
        return [x, recon]

    def loss_function(self, *args) -> dict:
        x_flat = torch.flatten(args[0], start_dim=1)
        recon_flat = torch.flatten(args[1], start_dim=1)
        recons_loss = F.mse_loss(x_flat, recon_flat)
        return {'loss': recons_loss}

    def training_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        results = self(sequences)
        train_loss_dict = self.loss_function(*results)
        self.log("train_loss", train_loss_dict['loss'], prog_bar=True, logger=True)
        return train_loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        results = self(sequences)
        val_loss_dict = self.loss_function(*results)
        self.log("val_loss", val_loss_dict['loss'], prog_bar=True, logger=True)
        return val_loss_dict['loss']

    def test_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        results = self(sequences)
        test_loss_dict = self.loss_function(*results)
        self.log("test_loss", test_loss_dict['loss'], prog_bar=True, logger=True)
        return test_loss_dict['loss']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer),         # The scheduler instance
            "interval": "epoch",                               # The unit of the scheduler's step size
            "frequency": 10,                    # How many epochs/steps should pass between calls to `scheduler.step()`
            "monitor": "val_loss",              # Metric to monitor for scheduler
            "strict": True,                     # Enforce that "val_loss" is available when the scheduler is updated
            "name": 'LearningRateMonitor',      # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }


def create_ae(encoder_model, decoder_model,
              dir_path="configs/logs", verbose=False, monitor="val_loss", mode="min",
              num_epochs=100, gpus=1,
              validation_hook_batch=None, test_hook_batch=None):

    _model = AutoEncoder(encoder_model=encoder_model, decoder_model=decoder_model)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project='AutoEncoder', job_type='train')

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
        patience=10,
        verbose=verbose
    )

    callbacks = [checkpoint_callback, early_stop_callback]
    # If we pass set of validation samples, add their callbacks
    if validation_hook_batch is not None:
        pass

    # If we pass a set of test samples, add their callbacks
    if test_hook_batch is not None:
        # TODO
        pass

    _trainer = pl.Trainer(
        default_root_dir=dir_path,
        logger=wandb_logger,
        callbacks=callbacks,
        checkpoint_callback=True,
        max_epochs=num_epochs,
        check_val_every_n_epoch=10,
        gpus=gpus,
    )

    return _model, _trainer
