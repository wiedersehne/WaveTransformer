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

from WaveLSTM.custom_callbacks import waveLSTM, attention, survival
from WaveLSTM.models.base import WaveletBase
from WaveLSTM.modules.self_attentive_encoder import SelfAttentiveEncoder
from pycox.evaluation import EvalSurv
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
    def norm_time(self):
        return self._norm_time

    @norm_time.setter
    def norm_time(self, t):
        self._norm_time = t

    @property
    def test_time(self):
        return self._norm_time

    @test_time.setter
    def test_time(self, t):
        self._test_time = t

    def sort(self, x, c, t, k):
        argsort_t = torch.argsort(t)
        x_ = x[argsort_t, :, :]
        c_ = c[argsort_t, :]
        t_ = t[argsort_t]
        k_ = k[argsort_t]
        return (x, c, t, k), argsort_t

    def __init__(self,
                 encoder_config,
                 attention_config,
                 config):

        super().__init__()
        WaveletBase.__init__(self,
                             seq_length=encoder_config["seq_length"],
                             recursion_limit=encoder_config["recursion_limit"],
                             wavelet=encoder_config["wavelet"])

        self.save_hyperparameters()
        self.channels, self.seq_length = encoder_config["channels"], encoder_config["seq_length"]
        self.cna = config["CNA"]
        # Scaling time
        #       (in ASCAT dataloader the survival time "t" and age in "c" are already standardised and normalised resp)
        self._norm_time = 1                   # By default don't scale time
        self._test_time = 1                   # By default test on range [0, _test_time]

        # Encoder
        encoder_config["J"] = self.J
        encoder_config["pooled_width"] = self.masked_width
        self.a_encoder = SelfAttentiveEncoder(encoder_config=encoder_config,
                                              config=attention_config)
        if config["pre_trained"] is not False:
            print("Loading pre-trained self attentive encoder")
            self.a_encoder.load_state_dict(torch.load(config["pre_trained"]))

        # Survival model
        # xdim = self.encoder.recursion_limit * encoder_config["scale_embed_dim"]
        hidden_dim = 32                                     # Hidden dimension size inside ODE model
        lr = np.inf                                         # This learning rate isnt used - just caveat of imported code
        if self.cna is True:
            # Flattened multi-resolution embedding dimension + number of additional covariates
            c_dim = encoder_config["scale_embed_dim"] * attention_config["attention-hops"]  + 2
        elif self.cna == "avg":
            c_dim = 3
        else:
            c_dim = 2
        self.surv_model = ODESurvSingle(lr, c_dim, hidden_dim, device="gpu")     # 2 baseline covariates

    def forward(self, x: torch.tensor, c: torch.tensor, t: torch.tensor, k: torch.tensor):
        """
        x: count number alteration data
        c: additional covariates
        t: survival time
        k: survival outcome
        """
        assert x.dim() == 3
        t /= self.norm_time
        meta_data = {}

        # Whether we additionally include the CNA data
        if self.cna is True:
            # Input masking
            masked_inputs, masked_targets = self.sequence_mask(self.scale(x))
            meta_data.update({
                'scaled_masked_inputs': masked_inputs,
                'scaled_masked_targets': masked_targets,
            })

            # Attentively encode
            h, meta_data = self.a_encoder(masked_inputs, meta_data)  # [batch_size, attention-hops, scale_embed_dim]
            meta_data.update({"M": h})
            # Flatten multi-resolution embeddings
            h = h.view(h.size(0), -1)                                # [batch_size, attention-hops * scale_embed_dim]
        elif self.cna == "avg":
            h = torch.mean(x.view(x.size(0), -1), 1, keepdim=True)   # [batch_size, 1]
        else:
            pass

        # Survival
        X = torch.concat((h, c), dim=1) if self.cna else c
        loss_survival = self.surv_model(X.type(torch.float32).to(device),
                                        t.type(torch.float32).to(device),
                                        k.type(torch.float32).to(device)) / X.shape[0]

        return {"loss": loss_survival}, meta_data

    def training_step(self, batch, batch_idx):
        x, t, k = batch['feature'], batch['survival_time'], batch['survival_status']
        c = torch.stack((batch["days_since_birth"].to(device),
                        torch.tensor([1 if i == "male" else 0 for i in batch['sex']], device=device)),
                        dim=1)
        (_x, _c, _t, _k), _ = self.sort(x, c, t, k)
        losses, meta_data = self(_x, _c, _t, _k)
        self.log("train_loss", losses["loss"], prog_bar=True, logger=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        x, t, k = batch['feature'], batch['survival_time'], batch['survival_status']
        c = torch.stack((batch["days_since_birth"].to(device),
                        torch.tensor([1 if i == "male" else 0 for i in batch['sex']], device=device)),
                        dim=1)
        (_x, _c, _t, _k), _ = self.sort(x, c, t, k)
        losses, meta_data = self(_x, _c, _t, _k)
        self.log("val_loss", losses["loss"], prog_bar=True, logger=True)
        return losses["loss"]

    def test_step(self, batch, batch_idx):
        # Get data
        x, t, k = batch['feature'], batch['survival_time'], batch['survival_status']
        c = torch.stack((batch["days_since_birth"].to(device),
                        torch.tensor([1 if i == "male" else 0 for i in batch['sex']], device=device)),
                        dim=1)
        labels = batch["label"]

        # Sort batch by time (TODO: not sure why Dom did this, but i'll do too for consistency)
        (_x, _c, _t, _k), argsort_t = self.sort(x, c, t, k)
        _l = labels[argsort_t]

        # Test loss
        test_loss = self(_x, _c, _t, _k)[0]["loss"]
        self.log("test_loss", test_loss, prog_bar=True, logger=True)

        # Predict  #TODO - move to callbacks
        if self.cna is True:
            masked_inputs, _ = self.sequence_mask(self.scale(_x))
            _h, _ = self.a_encoder(masked_inputs, {})
            _h = _h.view(_h.size(0), -1)
        elif self.cna == "avg":
            _h = torch.mean(_x.view(x.size(0), -1), 1, keepdim=True)      # [batch_size, 1]

        _X = torch.concat((_h, _c), dim=1) if self.cna else _c
        _X = _X.detach().cpu().numpy()
        n_eval = 1000
        t_eval = np.linspace(0, self.test_time, n_eval)
        t_test = torch.tensor(np.concatenate([t_eval] * _X.shape[0], 0), dtype=torch.float32, device=device)
        X_test = torch.tensor(np.repeat(_X, [t_eval.size] * _X.shape[0], axis=0), device=device, dtype=torch.float32)    # TODO: no need to do this operation outside of torch
        pred = self.surv_model.predict(X_test, t_test).reshape((_X.shape[0], t_eval.size)).cpu().detach().numpy()

        surv = pd.DataFrame(np.transpose((1 - pred)), index=t_eval)

        _t = _t.cpu().detach().numpy()
        _k = _k.cpu().detach().numpy()
        _l = _l.cpu().detach().numpy()
        ev = EvalSurv(surv, _t, _k, censor_surv='km')

        # Plot
        #TODO: remove hard code cancer labels
        cancer_types = ['OV', 'GBM', 'KIRC', 'HNSC', 'LGG']  #
        for lbl in range(len(cancer_types)):
            idx_lbl = np.where(_l == lbl)[0]
            # for k, idx_lbl_k in enumerate([np.where(_k[idx_lbl] == 0)[0], np.where(_k[idx_lbl] != 0)[0]]):
            #     idx_lbl_k = idx_lbl_k[:40] if len(idx_lbl_k) > 40 else idx_lbl_k
            idx_lbl = idx_lbl[:5] if len(idx_lbl) > 5 else idx_lbl
            for i in range(len(idx_lbl)):
                ev[idx_lbl[i]].plot_surv()
                plt.title(f"Cancer {cancer_types[lbl]}")# and {'event' if k == 1 else 'Censored'}")      #  at normalised time {_t[i]:.2f},
                plt.ylim((0,1))
            plt.show()

        time_grid = np.linspace(_t.min(), 0.9 * _t.max(), 1000)
        print(ev.concordance_td())
        print(ev.integrated_brier_score(time_grid))
        print(ev.integrated_nbll(time_grid))

        return test_loss

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

def create_desurv(seq_length, channels,
                  use_CNA=True,
                  pre_trained=None,
                  hidden_size=256, layers=1, proj_size=0, scale_embed_dim=128, wavelet='haar', recursion_limit=None,
                  dir_path="logs", verbose=False, monitor="val_loss",
                  num_epochs=200, gpus=1,
                  validation_hook_batch=None, test_hook_batch=None,
                  project='WaveLSTM-aeDeSurv', run_id="null",
                  outfile="logs/desurv-output.pkl"
                  ):

    encoder_config = {"seq_length": seq_length,
                      "channels": channels,
                      "hidden_size": hidden_size,
                      "scale_embed_dim": scale_embed_dim,
                      "layers": layers,
                      "proj_size": proj_size,
                      "wavelet": wavelet,
                      "recursion_limit": recursion_limit
                      }
    attention_config = {"dropout": 0.0,
                        "attention-unit": 350,
                        "attention-hops": 1,
                        "penalization_coeff": 0.,
                        "real_hidden_size": proj_size if proj_size > 0 else hidden_size,  # Encoder's hidden size
                        }
    surv_config = {"CNA": use_CNA,
                   "pre_trained": pre_trained}

    _model = DeSurv(encoder_config=encoder_config,
                    attention_config=attention_config,
                    config=surv_config)
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
        min_delta=0.,
        patience=5,
        verbose=verbose
    )

    surv_metrics = survival.PerformanceMetrics(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    viz_embedding_callback = waveLSTM.ResolutionEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch,
    )

    viz_multi_res_embed = attention.MultiResolutionEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch,
    )

    viz_attention = attention.Attention(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    save_output = waveLSTM.SaveOutput(
        test_samples=test_hook_batch,
        file_path=outfile
    )

    callbacks = [#surv_metrics,
                 checkpoint_callback,
                 early_stop_callback,
                 # save_output
                 ]
    # if use_CNA:
    #     callbacks.append([viz_embedding_callback,
    #                       viz_multi_res_embed,
    #                       viz_attention,])

    _trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=num_epochs,
        check_val_every_n_epoch=2,
        gpus=gpus,
    )

    return _model, _trainer
