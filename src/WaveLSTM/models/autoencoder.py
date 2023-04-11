from abc import ABC
# Torch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# PyTorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger              # tracking tool
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import ptwt
import pywt

from WaveLSTM.modules.encoder import Encoder
from WaveLSTM.custom_callbacks import waveLSTM_callbacks
from WaveLSTM.custom_callbacks import ae_callbacks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(pl.LightningModule, ABC):

    def masked_loss(self, x):
        """
        Get the masked target sequence by reconstructing wavelet coefficients with the highest resolutions masked
             get IWT(alpha_1, ..., alpha_J, 0, ...)
        """
        assert x.dim() == 4, x.shape

        targets = []
        for j in range(self.encoder.recursion_limit):
            masked_target = torch.zeros_like(x, device=device)
            for c in range(x.shape[1]):
                for h in range(x.shape[2]):
                    full_bank = ptwt.wavedec(x[:, c, h, :], self.wavelet, mode='zero', level=self.encoder.max_level)
                    masked_bank = [alpha_i if i <= j else torch.zeros_like(alpha_i)
                                   for i, alpha_i in enumerate(full_bank)]
                    masked_target[:, c, h, :] = ptwt.waverec(masked_bank, self.wavelet)

            targets.append(masked_target)

        return targets

    def __init__(self, seq_length, strands, chromosomes,
                 hidden_size=256, layers=1, proj_size=0, scale_embed_dim=128,
                 wavelet="haar",
                 recursion_limit=None,):

        super().__init__()
        self.save_hyperparameters()
        self.wavelet = pywt.Wavelet(wavelet)
        self.chromosomes = chromosomes
        self.strands = strands

        # Encoder
        self.encoder = Encoder(seq_length=seq_length, strands=strands, chromosomes=chromosomes,
                               hidden_size=hidden_size, layers=layers, proj_size=proj_size,
                               scale_embed_dim=scale_embed_dim,
                               wavelet=wavelet, recursion_limit=recursion_limit)

        # Decoder:
        #     Get the dimension of each detail space
        self.decoder_bank_lengths = \
            [t.shape[-1] for t in pywt.wavedec(np.zeros((1, self.strands, self.chromosomes, seq_length)),
                                               self.wavelet,
                                               level=self.encoder.max_level)]
        #    Connect scale embeddings to wavelet coefficient
        coeff_nets = [nn.Sequential(nn.ReLU(), nn.LazyLinear(out_features=length * strands * chromosomes, device=device)
                                    )
                      for length in self.decoder_bank_lengths[:self.encoder.recursion_limit]]
        self.coeff_nets = nn.ModuleList(coeff_nets)

    def forward(self, x: torch.tensor):

        if x.dim() == 3:
            x = x.unsqueeze(2)

        # Encode
        resolution_embeddings, meta_data = self.encoder(x)
        meta_data.update({'hidden': resolution_embeddings})

        # Decode
        predicted_bank = [torch.zeros((x.size(0), x.size(1), x.size(2), length), device=device)
                          for length in self.decoder_bank_lengths]
        autoencoder_outputs = []

        for j, scale_embed in enumerate(resolution_embeddings):
            # Auto-encoder
            predicted_bank[j] = self.coeff_nets[j](scale_embed).reshape(x.size(0), x.size(1), x.size(2), -1)
            partly_recon = torch.zeros(x.shape, device=device)
            for strand in range(self.strands):
                for chrom in range(self.chromosomes):
                    pred_bank_channel = [coeff[:, strand, chrom, :] for coeff in predicted_bank]
                    partly_recon[:, strand, chrom, :] = ptwt.waverec(pred_bank_channel, self.wavelet)

            # Record next output for target sequence
            autoencoder_outputs.append(partly_recon)

        meta_data.update({'filter_bank': predicted_bank,
                          'autoencoder_output': autoencoder_outputs,
                          'masked_target': self.masked_loss(x),
                          })

        return partly_recon, meta_data

    def loss_function(self, x, x_recon, meta_data, filter_noise=True) -> dict:
        if filter_noise:
            x = meta_data["masked_target"][-1]
        x = torch.flatten(x, start_dim=1)
        x_recon = torch.flatten(x_recon, start_dim=1)
        return {'loss': F.mse_loss(x, x_recon)}

    def training_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        recon, meta_results = self(sequences)
        train_loss_dict = self.loss_function(sequences, recon, meta_results)
        self.log("train_loss", train_loss_dict['loss'], prog_bar=True, logger=True)
        return train_loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        recon, meta_results = self(sequences)
        val_loss_dict = self.loss_function(sequences, recon, meta_results)
        self.log("val_loss", val_loss_dict['loss'], prog_bar=True, logger=True)
        return val_loss_dict['loss']

    def test_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        recon, meta_results = self(sequences)
        test_loss_dict = self.loss_function(sequences, recon, meta_results)
        self.log("test_loss", test_loss_dict['loss'], prog_bar=True, logger=True)
        return test_loss_dict['loss']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer),         # The scheduler instance
            "interval": "epoch",                               # The unit of the scheduler's step size
            "frequency": 5,                     # How many epochs/steps should pass between calls to `scheduler.step()`
            "monitor": "val_loss",              # Metric to monitor for scheduler
            "strict": True,                     # Enforce that "val_loss" is available when the scheduler is updated
            "name": 'LearningRateMonitor',      # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

def create_autoencoder(seq_length, strands, chromosomes,
                       hidden_size=256, layers=1, proj_size=0, scale_embed_dim=128,
                       wavelet='haar',
                       recursion_limit=None,
                       dir_path="configs/logs", verbose=False,  monitor="val_loss", mode="min",
                       num_epochs=20, gpus=1,
                       validation_hook_batch=None, test_hook_batch=None,
                       project='WaveLSTM-autoencoder', run_id="null"):

    _model = AutoEncoder(seq_length, strands, chromosomes,
                         hidden_size=hidden_size, layers=layers, proj_size=proj_size, scale_embed_dim=scale_embed_dim,
                         wavelet=wavelet, recursion_limit=recursion_limit,
                         )
    print(_model)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project=project,
                               name=run_id,
                               job_type='train',
                               save_dir=dir_path)

    # Make all callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_path + "/checkpoints",
        filename=run_id,
        verbose=verbose,
        monitor=monitor,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", mode="min",
        min_delta=0.01,
        patience=5,
        verbose=verbose
    )

    viz_embedding_callback = waveLSTM_callbacks.ViewEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    viz_prediction_callback = waveLSTM_callbacks.ViewSignal(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    save_output = waveLSTM_callbacks.SaveOutput(
        test_samples=test_hook_batch,
        file_path="../figs_and_demos/output.pkl"
    )

    ae_recon_callback = ae_callbacks.ViewRecurrentSignal(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    callbacks = [checkpoint_callback,
                 early_stop_callback,
                 viz_prediction_callback,
                 viz_embedding_callback,
                 save_output,
                 ae_recon_callback,
                 ]

    _trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=num_epochs,
        check_val_every_n_epoch=2,
        gpus=gpus,
    )

    return _model, _trainer
