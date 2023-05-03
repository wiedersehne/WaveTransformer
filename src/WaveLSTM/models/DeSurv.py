from abc import ABC
# Torch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# PyTorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import pandas as pd

from WaveLSTM.custom_callbacks import waveLSTM_callbacks
from WaveLSTM.custom_callbacks import ae_callbacks
from WaveLSTM.models.autoencoder import AutoEncoder
from WaveLSTM.modules.self_attentive_encoder import SelfAttentiveEncoder
from WaveLSTM.modules.encoder import Encoder
from pycox.evaluation import EvalSurv
from DeSurv.src.classes import ODESurvSingle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeSurv(pl.LightningModule, ABC):
    """ PyTorch lightning wrapper around DeSurv's single-risk ODE model
                    (ref: https://proceedings.mlr.press/v151/danks22a/danks22a.pdf)

    Modifications:
     - to be compatible with my own dataloaders
     - unvectorise predict step due to scaling issues with higher feature spaces
    """

    def __init__(self, seq_length, strands, chromosomes,
                 hidden_size=256, layers=1, proj_size=0, scale_embed_dim=128,
                 wavelet="haar",
                 recursion_limit=None):


        super().__init__()
        self.save_hyperparameters()
        self.base_loss_ratio = 0.001

        # Auto-encoder
        self.autoencoder = AutoEncoder(seq_length, strands, chromosomes,
                                       hidden_size=hidden_size, layers=layers, proj_size=proj_size,
                                       scale_embed_dim=scale_embed_dim,
                                       wavelet=wavelet,
                                       recursion_limit=recursion_limit)

        # Encoder
        # self.encoder = SelfAttentiveEncoder(encoder_config=encoder_config,
        #                                                    config=config)
        # self.encoder = Encoder(**encoder_config) if encoder_config is not None else None

        # Flatten
        self.flat_net = nn.Sequential(nn.Dropout(0.5),
                                      nn.Flatten()
                                      )

        # Survival model
        # xdim = self.encoder.recursion_limit * encoder_config["scale_embed_dim"]
        hidden_dim = 32                                     # Hidden dimension size inside ODE model
        lr = np.inf                                         # This learning rate isnt used - just caveat of imported code
        embed_dim = scale_embed_dim * recursion_limit       # Concat+Flattened resolution embedding dimension
        self.surv_model = ODESurvSingle(lr, embed_dim + 2, hidden_dim, device="gpu")     # 2 baseline covariates

    def forward(self, x: torch.tensor, c: torch.tensor=None, t: torch.tensor=None, k: torch.tensor=None):

        if x.dim() == 3:
            x = x.unsqueeze(2)

        if (c is not None) and (t is not None) and (k is not None):
            argsort_t = torch.argsort(t)
            x = x[argsort_t, :, :, :]

        loss_dict = {}

        # Encode
        x_recon, meta_data = self.autoencoder(x)
        loss_autoencoder = self.autoencoder.loss_function(x, x_recon=x_recon, meta_data=meta_data)
        loss_dict.update({'autoencoder_loss': loss_autoencoder["loss"]})

        loss_survival = 0
        if (c is not None) and (t is not None) and (k is not None):
            # Flatten
            h_ = self.flat_net(torch.stack(meta_data["hidden"], dim=1))

            # Survival
            c_ = c[argsort_t, :]
            t_ = t[argsort_t]
            k_ = k[argsort_t]
            X_ = torch.concat((h_, c_), dim=1)
            loss_survival = self.surv_model(X_.type(torch.float32), t_.type(torch.float32), k_.type(torch.float32))
            loss_dict.update({'DeSurv_loss': loss_survival})

        # Combination of decaying autoencoder loss and weighted survival loss
        decay1 = np.max((1, 5*(1-0.05)**self.current_epoch))
        loss_dict.update({'loss': (decay1 * loss_autoencoder["loss"]) +
                                  (self.base_loss_ratio * loss_survival)})

        return loss_dict, meta_data

    # def predict_step(self, batch, batch_idx):
    #
    #     # Signal inputs
    #     x = batch['feature']
    #     if x.dim() == 3:
    #         x = x.unsqueeze(2)
    #     x, meta_results = self.encoder(x)
    #     x = torch.stack(x, dim=1)
    #     x_test = self.drop(self.flat(x)).detach().cpu().numpy()
    #
    #     # Baseline covariates
    #
    #     t_test = batch['survival_time'].detach().cpu().numpy() #/ self.t_train_max
    #     e_test = batch['survival_status'].detach().cpu().numpy()
    #     lbl_test = batch['label'].detach().cpu().numpy()
    #
    #     argsortttest = np.argsort(t_test)
    #     t_test = t_test[argsortttest]
    #     e_test = e_test[argsortttest]
    #     x_test = x_test[argsortttest, :]
    #
    #     n_eval = 3000
    #     t_eval = np.linspace(0, self.t_n_test_max, n_eval)
    #
    #     surv = []
    #     for sample_idx in range(x_test.shape[0]):
    #         # t_ = torch.tensor(np.concatenate([t_eval] * x_test.shape[0], 0), dtype=torch.float32)
    #         t_ = torch.tensor(t_eval, dtype=torch.float32)
    #         # x_ = torch.tensor(np.repeat(x_test, [t_eval.size] * x_test.shape[0], axis=0), dtype=torch.float32)
    #         x_ = torch.tensor(np.repeat(x_test[[sample_idx], :], [t_eval.size], axis=0), dtype=torch.float32)
    #
    #         surv.append(pd.DataFrame(np.transpose(
    #             (1 - self.surv_model.predict(x_, t_).reshape((1, t_eval.size))).cpu().detach().numpy()),
    #             index=t_eval))
    #
    #     surv = pd.concat(surv, ignore_index=True, axis=1)
    #
    #     return {"surv": surv,
    #             "t_test": t_test,
    #             "e_test": e_test,
    #             "lbl_test": lbl_test}

    def training_step(self, batch, batch_idx):
        x, t, k = batch['feature'], batch['survival_time'], batch['survival_status']
        c = torch.stack((batch["days_since_birth"].to(device),
                        torch.tensor([1 if i == "male" else 0 for i in batch['sex']], device=device)),
                        dim=1)
        losses, meta_data = self(x, c, t, k)
        self.log("train_loss", losses["loss"], prog_bar=True, logger=True)
        self.log("train_recon_loss", losses["autoencoder_loss"], prog_bar=True, logger=True)
        self.log("train_surv_loss", losses["DeSurv_loss"], prog_bar=True, logger=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        x, t, k = batch['feature'], batch['survival_time'], batch['survival_status']
        c = torch.stack((batch["days_since_birth"].to(device),
                        torch.tensor([1 if i == "male" else 0 for i in batch['sex']], device=device)),
                        dim=1)
        losses, meta_data = self(x, c, t, k)
        self.log("val_loss", losses["loss"], prog_bar=True, logger=True)
        self.log("val_recon_loss", losses["autoencoder_loss"], prog_bar=True, logger=True)
        self.log("val_surv_loss", losses["DeSurv_loss"], prog_bar=True, logger=True)
        return losses["loss"]

    def test_step(self, batch, batch_idx):
        x, t, k = batch['feature'], batch['survival_time'], batch['survival_status']
        c = torch.stack((batch["days_since_birth"].to(device),
                        torch.tensor([1 if i == "male" else 0 for i in batch['sex']], device=device)),
                        dim=1)
        losses, meta_data = self(x, c, t, k)
        self.log("test_loss", losses["loss"], prog_bar=True, logger=True)
        self.log("test_recon_loss", losses["autoencoder_loss"], prog_bar=True, logger=True)
        self.log("test_surv_loss", losses["DeSurv_loss"], prog_bar=True, logger=True)
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer),         # The scheduler instance
            "interval": "epoch",                               # The unit of the scheduler's step size
            "frequency": 2,                     # How many epochs/steps should pass between calls to `scheduler.step()`
            "monitor": "val_loss",              # Metric to monitor for scheduler
            "strict": True,                     # Enforce that "val_loss" is available when the scheduler is updated
            "name": 'LearningRateMonitor',      # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

def create_desurv(seq_length, strands, chromosomes,
                  hidden_size=256, layers=1, proj_size=0, scale_embed_dim=128,
                  wavelet='haar',
                  recursion_limit=None,
                  dir_path="configs/logs", verbose=False, monitor="val_loss", mode="min",
                  num_epochs=200, gpus=1,
                  validation_hook_batch=None, test_hook_batch=None,
                  project='WaveLSTM-aeDeSurv', run_id="null"):

    _model = DeSurv(seq_length, strands, chromosomes,
                    hidden_size=hidden_size, layers=layers, proj_size=proj_size, scale_embed_dim=scale_embed_dim,
                    wavelet=wavelet, recursion_limit=recursion_limit)
    print(_model)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project=project,
                               name=run_id,
                               job_type='train',
                               save_dir=dir_path)

    # Make all callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_path + "/checkpoints",
        filename=f"{project}_{run_id}",
        verbose=verbose,
        monitor=monitor,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", mode="min",
        min_delta=0,
        patience=20,
        verbose=verbose
    )

    viz_embedding_callback = waveLSTM_callbacks.ViewEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )


    ae_recon_callback = ae_callbacks.ViewRecurrentSignal(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    callbacks = [checkpoint_callback,
                 early_stop_callback,
                 viz_embedding_callback,
                 ae_recon_callback
                 ]

    _trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=num_epochs,
        check_val_every_n_epoch=2,
        gpus=gpus,
    )

    return _model, _trainer
