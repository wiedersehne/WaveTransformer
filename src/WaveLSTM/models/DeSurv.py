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
from WaveLSTM.models.base import WaveletBase
from WaveLSTM.modules.self_attentive_encoder import SelfAttentiveEncoder
from DeSurv.src.classes import ODESurvSingle

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeSurv(pl.LightningModule, ABC, WaveletBase):
    """ PyTorch lightning wrapper around DeSurv's single-risk ODE model
                    (ref: https://proceedings.mlr.press/v151/danks22a/danks22a.pdf)

    Modifications:
     - to be compatible with my own dataloaders
     - unvectorise predict step due to scaling issues with higher feature spaces
     - refactoring
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

    def __init__(self,
                 encoder_config,
                 attention_config,
                 surv_config):

        super().__init__()
        WaveletBase.__init__(self,
                             seq_length=encoder_config["seq_length"],
                             recursion_limit=encoder_config["recursion_limit"],
                             wavelet=encoder_config["wavelet"])

        self.save_hyperparameters()
        self.channels, self.seq_length = encoder_config["channels"], encoder_config["seq_length"]
        self.encoder_type = surv_config["encoder_type"].lower()
        # Scaling time
        self._norm_time = 1                   # By default don't scale time
        self._test_time = 1                   # By default test on range [0, _test_time]
        # Penalisations
        self.diversity_coef = surv_config["diversity_coef"]
        self.weight_decay = surv_config["weight_decay"]

        # Encoder
        # Wavelet encoder
        if self.encoder_type == "wavelstm":
            encoder_config["J"] = self.J
            encoder_config["pooled_width"] = self.masked_width
            self.encoder = SelfAttentiveEncoder(encoder_config=encoder_config,
                                                config=attention_config)
        # CNN encoder
        if self.encoder_type == "cnn":
            # Same architecture as rcCAE. See: https://github.com/zhyu-lab/rccae/blob/main/cae/autoencoder.py
            # We reduce the width of the network by a factor of 4 to reduce overfitting.
            k_size = 7
            net_size = 4 # 4 gives the same encoder network as rcCAE (with chromosomes as channels)
            self.encoder = nn.Sequential(
                nn.Conv1d(self.channels, 32*net_size, k_size, stride=1),
                nn.LeakyReLU(),
                nn.Conv1d(32*net_size, 16*net_size, k_size, stride=1),
                nn.LeakyReLU(),
                nn.Conv1d(16*net_size, 8*net_size, k_size, stride=1),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.LazyLinear(out_features=encoder_config["scale_embed_dim"])
            )
        if self.encoder_type == "lstm":
            self.encoder = nn.LSTM(input_size=self.channels,
                                   hidden_size=encoder_config["hidden_size"],
                                   proj_size=encoder_config["scale_embed_dim"],
                                   num_layers=encoder_config["layers"],
                                   bidirectional=False,
                                   batch_first=True)
            self.encoder_outlayer = nn.LazyLinear(out_features=encoder_config["scale_embed_dim"])

        # Survival model
        hidden_dim = 32  # Hidden dimension size inside ODE model
        lr = np.inf                                          # This learning rate isnt used - just caveat of imported code
        if self.encoder_type == "wavelstm":
            # Flattened multi-resolution embedding dimension + number of additional covariates
            c_dim = encoder_config["scale_embed_dim"] * attention_config["attention-hops"]  + 2
        elif self.encoder_type in ["average", "avg"]:
            c_dim = 3
        elif self.encoder_type in ["cnn", "lstm"]:
            c_dim = encoder_config["scale_embed_dim"] + 2
        elif self.encoder_type in ["none"]:
            c_dim = 2
        else:
            raise NotImplementedError
        self.surv_model = ODESurvSingle(lr, c_dim, hidden_dim, device="gpu")     # 2 baseline covariates

    def forward(self, x: torch.tensor, c: torch.tensor, t: torch.tensor, k: torch.tensor):
        """
        Note: Due to how De-Surv is coded, we also have to return the loss in the forward def. This unfortunately means
                we must input survival time `t' into the forward call making it impossible to use this on test data
                where true survival time is not known. Instead you may need to overload the predict_step() function in
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
            h, meta_data = self.encoder(masked_inputs, meta_data)  # h: [batch_size, attention-hops, scale_embed_dim]
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

    def loss(self, batch: dict, batch_idx: int):
        x, t, k = batch['feature'], batch['survival_time'], batch['survival_status']
        c = torch.stack((batch["days_since_birth"].to(device),
                         torch.tensor([1 if i == "male" else 0 for i in batch['sex']], device=device)),
                        dim=1)
        losses, meta_data = self(x, c, t, k)
        loss = losses["loss"]

        # Calculate diversity penalization
        #   promotes diversity between hops if penality_coef > 0,
        #   promotes diversity within hops r==1 and < 0
        if "attention" in meta_data.keys() and self.diversity_coef != 0:
            atn = meta_data["attention"]            # [bsz, rhops, J]
            atn_t = torch.transpose(atn, 1, 2).contiguous()
            eye = torch.stack([torch.eye(atn.shape[1]) for _ in range(atn.shape[0])], dim=0)
            p = torch.norm(torch.bmm(atn, atn_t) - eye.to(device))
            loss += self.diversity_coef * p

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
            h, pred_meta_data = self.encoder(masked_inputs, pred_meta_data)  # [batch_size, attention-hops, scale_embed_dim]
            # Flatten multi-resolution embeddings
            h = h.view(h.size(0), -1)                               # [batch_size, attention-hops * scale_embed_dim]
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
        # X_test_Dom = torch.tensor(np.repeat(X.cpu(), [t_eval.size] * X.shape[0], axis=0), device=device,
        #                       dtype=torch.float32)  # TODO: keep inside torch
        X_test = X.repeat_interleave(t_eval.size, 0).to(device, torch.float32)

        # Cannot make all predictions at once due to memory constraints
        pred_batch = 16382                                                        # Predict in batches of `pred_batch`
        pred = []
        for X_test_batched, t_test_batched in zip(torch.split(X_test, pred_batch), torch.split(t_test, pred_batch)):
            pred.append(self.surv_model.predict(X_test_batched, t_test_batched))
        pred = torch.concat(pred)
        pred = pred.reshape((X.shape[0], t_eval.size)).cpu().detach().numpy()

        return pred, pred_meta_data

    def configure_optimizers(self):
        """ For survival model have a different scheduler for the encoder parameters and the task head (DeSurv) params"""
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)
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

def create_desurv(cancers, seq_length, channels,
                  encoder_type="waveLSTM",
                  J=None,
                  r_hops=10,
                  hidden_size=32,
                  h_proj=1,
                  D=1,
                  weight_decay=0,
                  dir_path="logs", verbose=False, monitor="val_loss",
                  num_epochs=200, gpus=1,
                  validation_hook_batch=None, test_hook_batch=None,
                  project='WaveLSTM-aeDeSurv', run_id="null",
                  outfile="logs/desurv-output.pkl",
                  ):

    encoder_config = {"seq_length": seq_length,
                      "channels": channels,
                      "hidden_size": hidden_size,                 # Dimension of cell state
                      "proj_size": h_proj, # Dimension of hidden cell state
                      "scale_embed_dim": D, # Dimension of relolution and multi-resolution embeddings
                      "layers": 1,  # Number of ConvLSTM layers
                      "kernel_size": 1, # Size of convolutional kernel inside ConvLSTMcell
                      "wavelet": 'haar',
                      "recursion_limit": J
                      }
    attention_config = {"dropout_embeddings": 0,
                        "attention-unit": 350,
                        "attention-hops": r_hops,
                        "real_hidden_size": h_proj if h_proj > 0 else hidden_size,  # Encoder's hidden size
                        }
    surv_config = {"encoder_type": encoder_type,
                   "diversity_coef": 0.0,
                   "weight_decay": weight_decay
                   }

    _model = DeSurv(encoder_config=encoder_config,
                    attention_config=attention_config,
                    surv_config=surv_config)
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
        min_delta=0,
        patience=1,
        verbose=verbose
    )

    label_dictionary = {key: val for key, val in zip([i for i in range(len(cancers))], cancers)}
    surv_metrics = survival.PerformanceMetrics(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    viz_KM = survival.KaplanMeier(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch,
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
    if encoder_type.lower() == "wavelstm":
        viz_embedding_callback = waveLSTM.ResolutionEmbedding(
            val_samples=validation_hook_batch,
            test_samples=test_hook_batch,
            label_dictionary=label_dictionary
        )

        viz_multi_res_embed = attention.MultiResolutionEmbedding(
            val_samples=validation_hook_batch,
            test_samples=test_hook_batch,
            label_dictionary=label_dictionary
        )

        viz_attention = attention.Attention(
            val_samples=validation_hook_batch,
            test_samples=test_hook_batch,
            label_dictionary=label_dictionary
        )

        save_output = waveLSTM.SaveOutput(
            test_samples=test_hook_batch,
            file_path=outfile
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
        max_epochs=num_epochs,
        log_every_n_steps=10,
        check_val_every_n_epoch=3,
        gpus=gpus,
    )

    return _model, _trainer
