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
# Modules
from WaveLSTM.models.base import WaveletBase as SourceSeparation
from WaveLSTM.modules.self_attentive_encoder import SelfAttentiveEncoder
# Callbacks
from WaveLSTM.custom_callbacks import waveLSTM
from WaveLSTM.custom_callbacks import attention
from WaveLSTM.custom_callbacks import autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentiveAutoEncoder(pl.LightningModule, ABC, SourceSeparation):

    def get_conv_shape(self, width, kernel_size, padding=0, stride=1):
        return int( (( width - kernel_size + (2 * padding)) / stride ) + 1 )

    def __init__(self, input_size, input_channels, config, pool_targets=False):
        """
        """
        super().__init__()
        self.save_hyperparameters()

        # Data
        self.input_channels = input_channels
        self.input_size = input_size
        # Reconstruct in averaged pooled domain, or original feature space
        self.pool_targets = pool_targets


        # Encoder
        SourceSeparation.__init__(self,
                             input_size=self.input_size,
                             recursion_limit=config.encoder.waveLSTM.J,
                             wavelet=config.encoder.waveLSTM.wavelet)
        self.encoder = SelfAttentiveEncoder(input_size=self.masked_width,
                                            input_channels=self.input_channels,
                                            D=config.encoder.base.D,
                                            **config.attention, **config.encoder.waveLSTM,
                                            )

        # Decoder:
        decoder_method=  config.decoder.base.method.lower()
        self.target_width = self.masked_width if self.pool_targets else self.input_size
        # Convolutional:  Decode from the multi-resolution embedding (M) using a transposed convolutional neural network
        if decoder_method == "rccae":
            k_size = 7
            stride = 1
            # Recursively work backwards, to find what the linear layer's output dim should be
            w_cnn = self.target_width * self.input_channels
            for cnn_layer in range(3):
                w_cnn = self.get_conv_shape(w_cnn, k_size, stride=stride)
            self.decoder = nn.Sequential(
                torch.nn.Flatten(start_dim=1),
                nn.Linear(config.encoder.base.D * config.attention.r_hops, 32 * w_cnn),
                torch.nn.Unflatten(1, (32, w_cnn)),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(32, 64, k_size, stride=stride),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(64, 128, k_size, stride=stride),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(128, 1, k_size, stride=stride),
                torch.nn.Flatten(start_dim=1),
                torch.nn.Unflatten(1, (self.input_channels, self.target_width))
            )
        elif decoder_method == "fc":
            # Fully connected
            nfc = 256
            self.decoder = nn.Sequential(
                torch.nn.Flatten(start_dim=1),
                nn.Linear(config.encoder.base.D * config.attention.r_hops, nfc),
                nn.ReLU(),
                nn.Linear(nfc, self.target_width * self.input_channels),
                nn.Unflatten(1, (self.input_channels, self.target_width)),
            )
        else:
            raise NotImplementedError

    def forward(self, x: torch.tensor):
        """
        """
        assert x.dim() == 3

        # Input masking. Perform IWT twice
        # - first for if we want to reconstruct with filterered targets (see loss: default False)
        _, masked_targets = self.sequence_mask(x, pool_targets=self.pool_targets)
        # - second for the input sequence for the waveLSTM encoder
        scaled_masked_inputs, _ = self.sequence_mask(self.scale(x))
        meta_data = {
            'masked_inputs': scaled_masked_inputs,
            'masked_targets': masked_targets,
        }

        # Attentively encode
        M, meta_data = self.encoder(scaled_masked_inputs, meta_data)
        meta_data.update({"M": M})                                 # [batch_size, attention-hops, resolution_embed_size]

        # Decode
        recon = self.decoder(M)                                    # [batch_size, channels, width]
        meta_data.update({'masked_predictions': [recon],
                          })

        return recon, meta_data

    def loss(self, batch, batch_idx, filter=True) -> dict:
        recon, meta_results = self(batch['feature'])
        target = meta_results["masked_targets"][-1] if filter else batch['feature']
        if self.pool_targets is False:
            target = self.scale(target)
        return {'loss': F.mse_loss(torch.flatten(recon, start_dim=1),
                                   torch.flatten(target, start_dim=1))
                }

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

def create_sa_autoencoder(data_module, test_data, val_data, cfg,
                  dir_path="logs",
                  gpus=1,
pool_targets=False,
                          ):
    # Data parameters
    labels = data_module.label_encoder.classes_
    W=data_module.W               # Signal length
    C=data_module.C             # Input channels

    _model = AttentiveAutoEncoder(input_size=W,  input_channels=C, config=cfg, pool_targets=pool_targets)
    if cfg.experiment.verbose:
        print(_model)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project=cfg.experiment.project_name,
                               name=cfg.experiment.run_id,
                               job_type='train',
                               save_dir=dir_path)

    # Make all callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_path + "/checkpoints",
        filename=cfg.experiment.run_id,
        verbose=cfg.experiment.verbose,
        monitor="val_loss",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", mode="min",
        min_delta=0.00,
        patience=3,
        verbose=cfg.experiment.verbose,
    )

    viz_res_embedding = waveLSTM.ResolutionEmbedding(
        val_samples=val_data,
        test_samples=test_data
    )

    viz_multi_res_embed = attention.MultiResolutionEmbedding(
        val_samples=val_data,
        test_samples=test_data
    )

    viz_attention = attention.Attention(
        val_samples=val_data,
        test_samples=test_data
    )

    viz_reconstruction = autoencoder.Reconstruction(
        val_samples=val_data,
        test_samples=test_data
    )

    save_output = waveLSTM.SaveOutput(
        test_samples=test_data,
        file_path=cfg.experiment.save_file
    )


    callbacks = [checkpoint_callback,
                 early_stop_callback,
                 viz_res_embedding,
                 viz_multi_res_embed,
                 viz_attention,
                 viz_reconstruction,
                 save_output,
                 ]

    _trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=cfg.experiment.num_epochs,
        check_val_every_n_epoch=5,
        log_every_n_steps=2,
        devices=gpus,
    )

    return _model, _trainer
