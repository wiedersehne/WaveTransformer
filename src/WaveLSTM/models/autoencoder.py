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

from WaveLSTM.models.base import WaveletBase
from WaveLSTM.modules.encoder import Encoder
from WaveLSTM.custom_callbacks import waveLSTM
from WaveLSTM.custom_callbacks import autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(pl.LightningModule, ABC, WaveletBase):

    def __init__(self, seq_length, channels,
                 hidden_size=256, layers=1, proj_size=0, scale_embed_dim=128,
                 wavelet="haar",
                 recursion_limit=None,
                 ):

        super().__init__()
        WaveletBase.__init__(self, seq_length=seq_length, recursion_limit=recursion_limit, wavelet=wavelet)

        self.save_hyperparameters()
        # self.wavelet = pywt.Wavelet(wavelet)
        self.channels, self.seq_length = channels, seq_length

        # Encoder
        self.encoder = Encoder(seq_length=seq_length, channels=channels, pooled_width=self.masked_width, J=self.J,
                               hidden_size=hidden_size, layers=layers, proj_size=proj_size,
                               scale_embed_dim=scale_embed_dim,
                               wavelet=wavelet, recursion_limit=recursion_limit)

        # Decoder:
        #    Connect scale embeddings to wavelet coefficient
        coeff_nets = [nn.Sequential(nn.LazyLinear(out_features=128, device=device),
                                    nn.ReLU(),
                                    nn.Linear(in_features=128, out_features=256, device=device),
                                    nn.ReLU(),
                                    nn.LazyLinear(out_features=length * channels, device=device)
                                    )
                      for length in self.alpha_lengths[:self.encoder.J]]
        self.coeff_nets = nn.ModuleList(coeff_nets)

    def forward(self, x: torch.tensor):

        assert x.dim() == 3

        # Input masking
        _, masked_targets = self.sequence_mask(x)

        scaled_masked_inputs, _ = self.sequence_mask(self.scale(x))
        meta_data = {
            'masked_inputs': scaled_masked_inputs,
            'masked_targets': masked_targets,
        }

        # Encode with wave-LSTM
        resolution_embeddings, meta_data = self.encoder(scaled_masked_inputs, meta_data)
        meta_data.update({
            'resolution_embeddings': resolution_embeddings
        })

        # Decode with fully connected networks, mapping resolution embeddings to each wavelet coefficient space, alpha_i
        filter_bank = [torch.zeros((x.size(0), x.size(1), length), device=device) for length in self.alpha_lengths]
        r_masked_predictions = []
        for j, scale_embed in enumerate(resolution_embeddings):
            filter_bank[j] = self.coeff_nets[j](scale_embed).reshape(x.size(0), x.size(1), -1)  # Predict coefficient
            r_masked_recon = torch.zeros(x.shape, device=device)                     # Memory-alloc for reconstructions
            for c in range(self.channels):                                           # Channelised reconstruction
                # Right-masked (IWT(alpha_1, ... alpha_j, 0,0...)
                r_pred_bank_channel = [coeff[:, c, :] for coeff in filter_bank]
                r_masked_recon[:, c, :] = ptwt.waverec(r_pred_bank_channel, self.wavelet)

            # Record next output for target sequence
            r_masked_predictions.append(r_masked_recon)

        meta_data.update({'filter_bank': filter_bank,
                          'masked_predictions': r_masked_predictions,
                          })

        return r_masked_recon, meta_data

    def loss(self, batch, batch_idx, filter=True) -> dict:
        recon, meta_results = self(batch['feature'])
        target = meta_results["masked_targets"][-1] if filter else batch['feature']
        return {'loss': F.mse_loss(torch.flatten(recon, start_dim=1),
                                   torch.flatten(self.scale(target), start_dim=1))}

    def training_step(self, batch, batch_idx):
        train_loss_dict = self.loss(batch, batch_idx)
        self.log("train_loss", train_loss_dict['loss'], prog_bar=True, logger=True)
        return train_loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        val_loss_dict = self.loss(batch, batch_idx)
        self.log("val_loss", val_loss_dict['loss'], prog_bar=True, logger=True)
        return val_loss_dict['loss']

    def test_step(self, batch, batch_idx):
        test_loss_dict = self.loss(batch, batch_idx)
        self.log("test_loss", test_loss_dict['loss'], prog_bar=True, logger=True)
        return test_loss_dict['loss']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer, verbose=True),         # The scheduler instance
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

def create_autoencoder(seq_length, channels,
                       hidden_size=256, layers=1, proj_size=0, scale_embed_dim=128,
                       wavelet='haar',
                       recursion_limit=None,
                       dir_path="logs", verbose=False,  monitor="val_loss",
                       num_epochs=20, gpus=1,
                       validation_hook_batch=None, test_hook_batch=None,
                       project='WaveLSTM-autoencoder', run_id="null",
                       outfile="logs/ae-output.pkl"
                       ):

    _model = AutoEncoder(seq_length, channels,
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
        monitor="val_loss",
        min_delta=0.001,
        patience=3,
        verbose=verbose
    )

    viz_res_embedding = waveLSTM.ResolutionEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    ae_recon_callback = autoencoder.Reconstruction(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    ae_rerecon_callback = autoencoder.RecurrentReconstruction(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    save_output = waveLSTM.SaveOutput(
        test_samples=test_hook_batch,
        file_path=outfile
    )

    callbacks = [checkpoint_callback,
                 early_stop_callback,
                 viz_res_embedding,
                 ae_recon_callback,
                 ae_rerecon_callback,
                 save_output,
                 ]

    _trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=num_epochs,
        check_val_every_n_epoch=2,
        log_every_n_steps=2,
        gpus=gpus,
    )

    return _model, _trainer