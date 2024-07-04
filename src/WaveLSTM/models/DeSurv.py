# Ref: https://proceedings.mlr.press/v151/danks22a/danks22a.pdf

from abc import ABC
# Torch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, LambdaLR
# PyTorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import pandas as pd
import logging

from WaveLSTM.custom_callbacks import waveLSTM, attention, survival
from WaveLSTM.models.base import WaveletBase as SourceSeparation
from WaveLSTM.modules.self_attentive_encoder import SelfAttentiveEncoder
from DeSurv.src.classes import ODESurvSingle

import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeSurv(pl.LightningModule, ABC):
    """ PyTorch lightning wrapper around DeSurv's single-risk ODE model
                    (ref: https://proceedings.mlr.press/v151/danks22a/danks22a.pdf)

    Inherits from WaveletBase - a class which performs discrete wavelet transform based input source separation

    Modifications to wrapped DeSurv code:
     - to be compatible with pytorch-lightning based dataloaders
     - unvectorise predict step due to reduce scaling issues with higher feature spaces
     - general refactoring
     - TODO: core DeSurv code's forward method returns loss, so our forward for this model does also
    """

    def __init__(self,
                 input_size,
                 input_channels,
                 time_scale,
                 config):
        """
        """
        super().__init__()
        self.save_hyperparameters()

        # Data
        self.input_size = input_size
        self.input_channels = input_channels
        self.time_scale = time_scale            # Internal time scaling

        # Initialise encoder
        self.encoder_type = config.encoder.base.method.lower()
        if self.encoder_type == "wavelstm":
            # WaveLSTM encoder
            logging.info(f"Using WaveLSTM encoder for Counter Number Alteration encodings")
            self.source_separation_layer = SourceSeparation(input_size=self.input_size,
                                                            input_channels=self.input_channels,
                                                            recursion_limit= config.encoder.waveLSTM.J,
                                                            wavelet= config.encoder.waveLSTM.wavelet)
            self.encoder = SelfAttentiveEncoder(input_size=self.source_separation_layer.masked_width,
                                                input_channels=self.input_channels,
                                                D=config.encoder.base.D,
                                                **config.encoder.waveLSTM,
                                                )
             # Number of encoded inputs to DeSurv
            c_dim = (config.encoder.base.D * config.encoder.waveLSTM.r_hops) + 2 # Number of encoded inputs to DeSurv
        elif self.encoder_type in ["cnn", "multiscale_cnn"]:
            # CNN encoder
            logging.info(f"Using CNN encoder for Counter Number Alteration encodings")
            # Same architecture as rcCAE. See: https://github.com/zhyu-lab/rccae/blob/main/cae/autoencoder.py
            k_size = 7
            self.encoder = nn.Sequential(
                nn.Conv1d(self.input_channels, 128, kernel_size=7, stride=1),
                nn.LeakyReLU(),
                nn.Conv1d(128, 64, kernel_size=7, stride=1),
                nn.LeakyReLU(),
                nn.Conv1d(64, 32, kernel_size=7, stride=1),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.LazyLinear(out_features=config.encoder.base.D)
            )
            c_dim = config.encoder.base.D + 2# Number of encoded inputs to DeSurv
        elif self.encoder_type == "lstm":
            # Uni-directional LSTM
            logging.info(f"Using uni-directional LSTM encoder for Counter Number Alteration encodings")
            self.encoder = nn.LSTM(input_size=self.input_channels,
                                   hidden_size=config.encoder.lstm.hidden_size,
                                   proj_size=config.encoder.lstm.proj_size,
                                   num_layers=config.encoder.lstm.layers,
                                   bidirectional=False,
                                   batch_first=True)
            self.encoder_outlayer = nn.LazyLinear(out_features=config.encoder.base.D)
            c_dim = config.encoder.base.D + 2         # Number of encoded inputs to DeSurv
        elif self.encoder_type in ["average", "avg"]:
            # Average copy number
            logging.info(f"Using average of Counter Number Alteration")
            c_dim = 3   # Number of encoded inputs to DeSurv
        elif self.encoder_type in ["none"]:
            # Not using Copy Number Alterations
            logging.info(f"Not using Counter Number Alterations")
            c_dim = 2   # Do not use CNA data
        else:
            logging.warning(f"Encoder type {config.encoder.base.method} is not supported")
            raise NotImplementedError

        # De-noising dropout on encoded latent variables
        self.dropout = nn.Dropout(config.encoder.base.dropout) if config.encoder.base.dropout is not None else None

        # Survival model
        hidden_dim = config.DeSurv.hidden  # Hidden dimension size inside ODE model
        lr = np.inf                        # This learning rate isnt used - just consequence of DeSurv's code structure
        self.surv_model = ODESurvSingle(lr, c_dim, hidden_dim, device="gpu")

    def forward(self, x: torch.tensor, c: torch.tensor, t: torch.tensor, k: torch.tensor):
        """
        Note: Due to how De-Surv is coded, we also have to return the loss in the forward def. This unfortunately means
                we must input survival time `t' into the forward call making it impossible to use this on test data
                where true survival time is not known. Instead we need to overload the predict_step() function in
                this case.

        x: count number alteration data
        c: additional covariates
        t: survival time
        k: survival outcome
        """
        assert x.dim() == 3
        t /= self.time_scale
        meta_data = {}

        # Whether we additionally include the CNA data
        if self.encoder_type == "wavelstm":
            # Input masking
            masked_inputs, masked_targets = self.source_separation_layer(x)
            meta_data.update({
                'masked_inputs': [m_i.detach().cpu().numpy() for m_i in masked_inputs],
                'masked_targets':[m_t.detach().cpu().numpy() for m_t in masked_targets],
            })
            # Attentively encode
            h, meta_data = self.encoder(masked_inputs, meta_data)  # h: [batch_size, attention-hops, resolution_embed_size]
            meta_data.update({"M": h.detach().cpu().numpy()})
            h = h.view(h.size(0), -1)                  # Flatten multi-resolution embeddings
            h = self.dropout(h) if self.dropout is not None else h
            X = torch.concat((h, c), dim=1)
        elif self.encoder_type == "cnn":
            h = self.encoder(x)
            h = self.dropout(h) if self.dropout is not None else h
            X = torch.concat((h, c), dim=1)
        elif self.encoder_type == "lstm":
            out, (hn, cn) = self.encoder(x.permute(0, 2, 1))
            h = self.encoder_outlayer(cn[-1, :, :])
            h = self.dropout(h) if self.dropout is not None else h
            # h = hn[-1, :, :]
            X = torch.concat((h, c), dim=1)
        elif self.encoder_type in ["average", "avg"]:
            h = torch.mean(x.view(x.size(0), -1), 1, keepdim=True)   # [batch_size, 1]
            X = torch.concat((h, c), dim=1)
        elif self.encoder_type in ["none"]:
            X = c
        else:
            raise NotImplementedError

        # Survival
        meta_data.update({"ode_input": X.detach().cpu().numpy()})
        loss_survival = self.surv_model(X.type(torch.float32).to(device),
                                        t.type(torch.float32).to(device),
                                        k.type(torch.float32).to(device))

        return {"loss": loss_survival / X.shape[0]}, meta_data

    def loss(self, batch: dict, batch_idx: int, diversity_coef=0):
        x = batch['CNA']
        c = batch["covariates"]
        t = batch['survival_time']
        k = batch['survival_status']
        losses, meta_data = self(x, c, t, k)
        loss = losses["loss"]

        # Calculate diversity penalization - note, this is not used in the paper
        #   promotes diversity between hops if r_hops > 1 and penality_coef > 0,
        if "attention" in meta_data.keys() and diversity_coef != 0:
            atn = meta_data["attention"]            # [bsz, rhops, J]
            atn_t = torch.transpose(atn, 1, 2).contiguous()
            eye = torch.stack([torch.eye(atn.shape[1]) for _ in range(atn.shape[0])], dim=0)
            p = torch.norm(torch.bmm(atn, atn_t) - eye.to(device))
            loss += diversity_coef * p

        return {"loss": loss}

    def training_step(self, batch: dict, batch_idx: int):
        losses = self.loss(batch, batch_idx)
        self.log("train_loss", losses["loss"], batch_size=batch["CNA"].shape[0],
                 prog_bar=True, logger=True, on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        losses = self.loss(batch, batch_idx)
        self.log("val_loss", losses["loss"], batch_size=batch["CNA"].shape[0],
                 prog_bar=True, logger=True)
        return losses["loss"]

    def test_step(self, batch: dict, batch_idx: int):
        losses = self.loss(batch, batch_idx)
        self.log("test_loss", losses["loss"], batch_size=batch["CNA"].shape[0],
                 prog_bar=True, logger=True)
        return losses["loss"]

    def predict(self, x: torch.tensor, c: torch.tensor, t: np.ndarray):
        """
        """
        t_eval = t / self.time_scale
        pred_meta_data = {}

        # Whether we additionally include the CNA data
        if self.encoder_type.lower() == "wavelstm":
            # Input masking and then attentively encode separated resolutions
            masked_inputs, _ = self.source_separation_layer(x)
            h, pred_meta_data = self.encoder(masked_inputs, pred_meta_data)  # [batch_size, attention-hops, resolution_embed_size]
            # Flatten multi-resolution embeddings
            h = h.view(h.size(0), -1)                               # [batch_size, attention-hops * resolution_embed_size]
            X = torch.concat((h, c), dim=1)
        elif self.encoder_type == "cnn":
            h = self.encoder(x)
            X = torch.concat((h, c), dim=1)
        elif self.encoder_type == "lstm":
            out, (hn, cn) = self.encoder(x.permute(0, 2, 1))
            h = self.encoder_outlayer(cn[-1, :, :])
            X = torch.concat((h, c), dim=1)
        elif self.encoder_type in ["average", "avg"]:
            h = torch.mean(x.view(x.size(0), -1), 1, keepdim=True)   # [batch_size, 1]
            X = torch.concat((h, c), dim=1)
        else:
            X = c

        # All x.size(0) * n_eval prediction inputs
        t_test = torch.tensor(np.concatenate([t_eval] * X.shape[0], 0), dtype=torch.float32, device=device)
        X_test = X.repeat_interleave(t_eval.size, 0).to(device, torch.float32)

        # Batched predict: Cannot make all predictions at once due to memory constraints
        pred_batch = 16382                                                        # Predict in batches
        pred = []
        for X_test_batched, t_test_batched in zip(torch.split(X_test, pred_batch), torch.split(t_test, pred_batch)):
            pred.append(self.surv_model.predict(X_test_batched, t_test_batched))
        pred = torch.concat(pred)
        pred = pred.reshape((X.shape[0], t_eval.size)).cpu().detach().numpy()

        return pred, pred_meta_data

    def configure_optimizers(self):
        """ For survival model have a different scheduler for the encoder parameters and the task head (DeSurv) params"""
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer, verbose=True, factor=0.5),         # The scheduler instance
            "interval": "epoch",                               # The unit of the scheduler's step size
            "frequency": 10,                     # How many epochs/steps should pass between calls to `scheduler.step()`
            "monitor": "val_loss",              # Metric to monitor for scheduler
            "strict": True,                     # Enforce that "val_loss" is available when the scheduler is updated
            "name": 'LearningRateMonitor',      # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

def create_desurv(data_module, cfg, time_scale=1, gpus=1):

    # Get validation and test hook batch
    val_data = next(iter(data_module.val_dataloader()))
    test_data = next(iter(data_module.test_dataloader()))

    # Create model
    _model = DeSurv(input_size=data_module.W ,
                    input_channels=data_module.C,
                    time_scale=time_scale,
                    config=cfg)
    logging.debug(_model)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project=cfg.experiment.project_name,
                               name=cfg.experiment.run_id,
                               job_type='train',
                               save_dir=cfg.experiment.output_dir)

    # Make all callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.experiment.output_dir}checkpoints/",
        filename=cfg.experiment.run_id,
        verbose=cfg.experiment.verbose,
        monitor="val_loss",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", mode="min",
        min_delta=0,
        patience=cfg.experiment.early_stopping,
        verbose=cfg.experiment.verbose
    )

    cancer_names = data_module.label_encoder.classes_
    label = data_module.label_encoder.transform(cancer_names)
    label_dictionary = {key: val for key, val in zip(label, cancer_names)}
    surv_metrics = survival.PerformanceMetrics(
        val_samples=val_data,
        test_samples=test_data
    )

    viz_KM = survival.KaplanMeier(
        val_samples=val_data,
        test_samples=test_data,
        label_dictionary=label_dictionary,
        group_by=["label"],
        error_bars=True,
        samples=True,
    )

    callbacks = [checkpoint_callback,
                 early_stop_callback,
                 surv_metrics,
                 viz_KM,
                 ]
    if cfg.encoder.base.method.lower() == "wavelstm":
        viz_embedding_callback = waveLSTM.ResolutionEmbedding(
            val_samples=val_data,
            test_samples=test_data,
            label_dictionary=label_dictionary
        )

        viz_multi_res_embed = attention.MultiResolutionEmbedding(
            val_samples=val_data,
            test_samples=test_data,
            label_dictionary=label_dictionary
        )

        viz_attention = attention.Attention(
            val_samples=val_data,
            test_samples=test_data,
            label_dictionary=label_dictionary
        )

        # save_output = waveLSTM.SaveOutput(
        #     test_samples=test_data,
        #     file_path=cfg.experiment.save_file
        # )

        # Add the wave-LSTM encoder callbacks
        callbacks += [viz_embedding_callback,
                      viz_multi_res_embed,
                      viz_attention,
                      # save_output
                      ]

    _trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=cfg.experiment.num_epochs,
        log_every_n_steps=10,
        check_val_every_n_epoch=2,
        devices=gpus,
    )

    return _model, _trainer
