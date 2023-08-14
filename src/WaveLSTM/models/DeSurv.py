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

from WaveLSTM.custom_callbacks import waveLSTM, attention, survival
from WaveLSTM.models.base import WaveletBase as SourceSeparation
from WaveLSTM.modules.self_attentive_encoder import SelfAttentiveEncoder
from DeSurv.src.classes import ODESurvSingle

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeSurv(pl.LightningModule, ABC, SourceSeparation):
    """ PyTorch lightning wrapper around DeSurv's single-risk ODE model
                    (ref: https://proceedings.mlr.press/v151/danks22a/danks22a.pdf)

    Inherits from WaveletBase - a class which performs discrete wavelet transform based input source separation

    Modifications to wrapped DeSurv code:
     - to be compatible with pytorch-lightning based dataloaders
     - unvectorise predict step due to reduce scaling issues with higher feature spaces
     - general refactoring
     - TODO: core DeSurv code's forward method returns loss, so our forward for this model does also
    """

    @property
    def time_scale(self):
        return self._time_scale

    @time_scale.setter
    def time_scale(self, t):
        self._time_scale = t

    @property
    def max_test_time(self):
        return self._test_time

    @max_test_time.setter
    def max_test_time(self, t):
        self._test_time = t

    def transform_time(self, t):
        if torch.is_tensor(t):
            return t / self.time_scale
        else:
            return t / self.time_scale

    def __init__(self, input_size,  input_channels,                    config ):
        """
        """
        super().__init__()
        self.save_hyperparameters()

        # Data
        self.input_channels = input_channels
        self.input_size = input_size
        # Scaling time
        self._norm_time = 1  # By default don't scale time
        self._test_time = 1  # By default test on range [0, _test_time]

        # Initialise encoder
        self.encoder_type = config.encoder.base.method.lower()
        if self.encoder_type == "wavelstm":
            # Wavelet encoder
            SourceSeparation.__init__(self,
                                 input_size=self.input_size,
                                 recursion_limit= config.encoder.waveLSTM.J,
                                 wavelet= config.encoder.waveLSTM.wavelet)
            self.encoder = SelfAttentiveEncoder(input_size=self.masked_width,
                                                input_channels=self.input_channels,
                                                D=config.encoder.base.D,
                                                **config.attention, **config.encoder.waveLSTM,
                                                )
             # Number of encoded inputs to DeSurv
            c_dim = (config.encoder.base.D * config.attention.r_hops)  + 2 # Number of encoded inputs to DeSurv
        elif self.encoder_type == "cnn":
            # CNN encoder
            # Same architecture as rcCAE. See: https://github.com/zhyu-lab/rccae/blob/main/cae/autoencoder.py
            k_size = 7
            self.encoder = nn.Sequential(
                nn.Conv1d(self.input_channels, 128, k_size=7, stride=1),
                nn.LeakyReLU(),
                nn.Conv1d(128, 64, k_size=7, stride=1),
                nn.LeakyReLU(),
                nn.Conv1d(64, 32, k_size=7, stride=1),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.LazyLinear(out_features=config.encoder.base.D)
            )
            c_dim = config.encoder.base.D + 2# Number of encoded inputs to DeSurv
        elif self.encoder_type== "lstm":
            # Uni-directional LSTM
            self.encoder = nn.LSTM(input_size=self.input_channels,
                                   hidden_size=config.encoder.lstm.hidden_size,
                                   proj_size=config.encoder.lstm.proj_size,
                                   num_layers=config.encoder.lstm.layers,
                                   bidirectional=False,
                                   batch_first=True)
            self.encoder_outlayer = nn.LazyLinear(out_features=config.encoder.base.D)
            c_dim = config.encoder.base.D + 2# Number of encoded inputs to DeSurv
        elif self.encoder_type in ["average", "avg"]:
            c_dim = 3# Number of encoded inputs to DeSurv
        elif self.encoder_type == None:
            c_dim = 2   # Do not use CNA data
        else:
            raise NotImplementedError

        # Survival model
        hidden_dim = 32  # Hidden dimension size inside ODE model
        lr = np.inf                                          # This learning rate isnt used - just caveat of imported code
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
        t = self.transform_time(t)
        meta_data = {}

        # Whether we additionally include the CNA data
        if self.encoder_type == "wavelstm":
            # Input masking
            masked_inputs, masked_targets = self.sequence_mask(self.scale(x))
            meta_data.update({
                'scaled_masked_inputs': masked_inputs,
                'scaled_masked_targets': masked_targets,
            })
            # Attentively encode
            h, meta_data = self.encoder(masked_inputs, meta_data)  # h: [batch_size, attention-hops, resolution_embed_size]
            meta_data.update({"M": h})
            h = h.view(h.size(0), -1)                  # Flatten multi-resolution embeddings
            X = torch.concat((h, c), dim=1)
        elif self.encoder_type == "cnn":
            h = self.encoder(self.scale(x))
            X = torch.concat((h, c), dim=1)
        elif self.encoder_type == "lstm":
            out, (hn, cn) = self.encoder(self.scale(x).permute(0, 2, 1))
            h = self.encoder_outlayer(cn[-1, :, :])
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
        meta_data.update({"ode_input": X})
        loss_survival = self.surv_model(X.type(torch.float32).to(device),
                                        t.type(torch.float32).to(device),
                                        k.type(torch.float32).to(device))

        return {"loss": loss_survival / X.shape[0]}, meta_data

    def loss(self, batch: dict, batch_idx: int, diversity_coef=0):
        x, t, k = batch['feature'], batch['survival_time'], batch['survival_status']
        c = torch.stack((batch["days_since_birth"].to(device),
                         torch.tensor([1 if i == "male" else 0 for i in batch['sex']], device=device)),
                        dim=1)
        losses, meta_data = self(x, c, t, k)
        loss = losses["loss"]

        # Calculate diversity penalization - this is not used in the accompanying paper
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
        self.log("train_loss", losses["loss"], batch_size=batch["feature"].shape[0],
                 prog_bar=True, logger=True, on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        losses = self.loss(batch, batch_idx)
        self.log("val_loss", losses["loss"], batch_size=batch["feature"].shape[0],
                 prog_bar=True, logger=True)
        return losses["loss"]

    def test_step(self, batch: dict, batch_idx: int):
        losses = self.loss(batch, batch_idx)
        self.log("test_loss", losses["loss"], batch_size=batch["feature"].shape[0],
                 prog_bar=True, logger=True)
        return losses["loss"]

    def predict(self, x: torch.tensor, c: torch.tensor, t_eval: np.ndarray):
        """
        """
        t_eval = self.transform_time(t_eval)
        pred_meta_data = {}

        # Whether we additionally include the CNA data
        if self.encoder_type == "wavelstm":
            # Input masking
            masked_inputs, _ = self.sequence_mask(self.scale(x))
            # Attentively encode
            h, pred_meta_data = self.encoder(masked_inputs, pred_meta_data)  # [batch_size, attention-hops, resolution_embed_size]
            # Flatten multi-resolution embeddings
            h = h.view(h.size(0), -1)                               # [batch_size, attention-hops * resolution_embed_size]
            X = torch.concat((h, c), dim=1)
        elif self.encoder_type == "cnn":
            h = self.encoder(self.scale(x))
            X = torch.concat((h, c), dim=1)
        elif self.encoder_type == "lstm":
            out, (hn, cn) = self.encoder(self.scale(x).permute(0, 2, 1))
            # h = hn[-1, :, :]
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

        # Cannot make all predictions at once due to memory constraints
        pred_batch = 16382                                                        # Predict in batches
        pred = []
        for X_test_batched, t_test_batched in zip(torch.split(X_test, pred_batch), torch.split(t_test, pred_batch)):
            pred.append(self.surv_model.predict(X_test_batched, t_test_batched))
        pred = torch.concat(pred)
        pred = pred.reshape((X.shape[0], t_eval.size)).cpu().detach().numpy()

        return pred, pred_meta_data

    def configure_optimizers(self):
        """ For survival model have a different scheduler for the encoder parameters and the task head (DeSurv) params"""
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=0)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer, verbose=True, factor=0.5),         # The scheduler instance
            "interval": "epoch",                               # The unit of the scheduler's step size
            "frequency": 3,                     # How many epochs/steps should pass between calls to `scheduler.step()`
            "monitor": "val_loss",              # Metric to monitor for scheduler
            "strict": True,                     # Enforce that "val_loss" is available when the scheduler is updated
            "name": 'LearningRateMonitor',      # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

def create_desurv(data_module, test_data, val_data, cfg,
                  dir_path="logs",
                  gpus=1,
                  ):

    # Data parameters
    labels = data_module.label_encoder.classes_
    W=data_module.W               # Signal length
    C=data_module.C             # Input channels

    # Create model
    _model = DeSurv(input_size=W,  input_channels=C,                    config=cfg)
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
        min_delta=0,
        patience=1,
        verbose=cfg.experiment.verbose
    )

    # TODO: add this to the data module to avoid bugs getting order wrong
    label_dictionary = {key: val for key, val in zip([i for i in range(len(labels))], labels)}
    print(label_dictionary)
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

        save_output = waveLSTM.SaveOutput(
            test_samples=test_data,
            file_path=cfg.experiment.save_file
        )

        # Add the wave-LSTM encoder callbacks
        callbacks += [viz_embedding_callback,
                      viz_multi_res_embed,
                      viz_attention,
                      save_output
                      ]

    _trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=cfg.experiment.num_epochs,
        log_every_n_steps=10,
        check_val_every_n_epoch=3,
        gpus=gpus,
    )

    return _model, _trainer
