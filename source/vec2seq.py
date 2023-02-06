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
from source.custom_callbacks.vec2seq_callbacks import *

import ptwt
import pywt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vec2Seq(pl.LightningModule, ABC):

    def fwt(self, x, wavelet):
        # Fast wavelet transform
        x_flat = torch.reshape(x, (x.size(0), -1))
        return ptwt.wavedec(x_flat, wavelet, mode='zero', level=self.max_level)

    @staticmethod
    def fc_network(out_features, hidden=128):
        return nn.LazyLinear(out_features=out_features, device=device)

    def __init__(self,
                 recurrent_net,
                 wavelet="haar",
                 auto_reccurent=False,
                 teacher_forcing_ratio=0.,
                 coarse_skip: int = 0,
                 recursion_limit=None,
                 ):

        assert coarse_skip >= 0

        super().__init__()
        self.save_hyperparameters()
        self.autorecurrent = auto_reccurent
        l = recurrent_net.L

        # Model
        self.teacher_forcing = teacher_forcing_ratio

        # Data

        # Wavelet
        self.wavelet = pywt.Wavelet(wavelet)
        # # Calculate the filter bank sizes, and recursion depth
        self.max_level = pywt.dwt_max_level(l, wavelet) - coarse_skip
        self.full_bank_lengths = [t.shape[1] for t in pywt.wavedec(np.zeros((1, l)), self.wavelet,
                                                                   level=self.max_level)]
        # Detail space lengths, then insert first approximation space length
        self.recursion_limit = len(self.full_bank_lengths) if recursion_limit is None \
            else np.min((recursion_limit, len(self.full_bank_lengths)))
        self.bank_lengths = self.full_bank_lengths[:recursion_limit]

        # Encoder/rec_net
        self.rec_net = recurrent_net

        # Connect LSTM output to wavelet coefficient
        self.rec_net.coeff_nets = [self.fc_network(out_features=length) for length in self.bank_lengths]

    def __str__(self):
        s = ''
        s += f'\nData'
        s += f'\n\t Data size (B, {self.rec_net.C}, {self.rec_net.H}, {self.rec_net.W})'
        s += f'\nRecurrent network'
        s += f'\n\t Wavelet "{self.wavelet.name}", which has decomposition length {self.wavelet.dec_len}'
        s += f'\n\t Full filter bank lengths {self.full_bank_lengths}'
        s += f'\n\t Recursion is over bank lengths {self.bank_lengths}, with recursion limit {self.recursion_limit}'
        if self.autorecurrent:
            s += f"\n\t Teacher forcing ratio={self.teacher_forcing}"
        s += str(self.rec_net)
        return s

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(path)

    def forward(self, x: torch.tensor, teacher_forcing: float, **kwargs):
        """
        x,                 features = [B, Strands, Chromosomes, Sequence Length]
        teacher_forcing,   the teacher forcing ratio for our RNN rec_net (used in training)
        """
        # Get ground truth target sequence
        true_bank = self.fwt(x.view(x.size(0), -1), wavelet=self.wavelet)

        # Partial fidelity ground truth for, removing the contribution of coefficients further down the recurrence
        true_masked_recons = [torch.zeros_like(x, device=device)]     # masked ground truth (zeroing higher spaces)
        for t in range(self.recursion_limit):
            _true_masked_bank = [AD if i <= t else torch.zeros_like(AD) for i, AD in enumerate(true_bank)]
            true_masked_recons.append(ptwt.waverec(_true_masked_bank, self.wavelet).reshape(x.shape))

        # Initialise hidden and cell states, and first LSTM input (usually this would be a <SOS> token in NLP)
        hidden, cell = self.rec_net.init_states(x, "learn")
        partly_recon = torch.zeros(x.shape, device=device)

        # Recurrent outputs:
        #  the approximation/detail space coefficients, which gets refined at each step
        #  the corresponding reconstructed features
        #  the hidden embedding at that scale
        predicted_bank = [torch.zeros((x.size(0), length), device=device) for length in self.full_bank_lengths]
        pred_masked_recons = []
        hidden_embedding = [hidden]

        def recursion(_residual, _hidden, _cell, _pred_bank, _t):
            """ in: residual signal, previous hidden and previous cell states
                out: tensor (predictions) and new hidden and cell states
            """
            coeff, (_hidden, _cell) = self.rec_net(_residual, _hidden, _cell, _t)
            _pred_bank[_t] = coeff

            _reconstruction = ptwt.waverec(_pred_bank, self.wavelet).reshape(_residual.shape)
            # print(f" in recursion {output.shape}, new hidden {_hidden.shape} and new cell {_cell.shape}")
            return _reconstruction, (_hidden, _cell), _pred_bank

        for t in range(self.recursion_limit):

            if self.autorecurrent:
                # TODO: detach output so gradients don't flow through? https://discuss.pytorch.org/t/correct-way-to-train-without-teacher-forcing/15508
                # Teacher forcing
                partly_recon = true_masked_recons[t] if random.random() < self.teacher_forcing else partly_recon
                partly_recon, (hidden, cell), predicted_bank = recursion(x - partly_recon,
                                                                         hidden, cell, predicted_bank, t)
            else:
                partly_recon, (hidden, cell), predicted_bank = recursion(x - true_masked_recons[t],
                                                                         hidden, cell, predicted_bank, t)

            # Record next output for target sequence
            pred_masked_recons.append(partly_recon)
            hidden_embedding.append(hidden)

        meta_data = {'true_bank': true_bank,
                     'filter_bank': predicted_bank,
                     'pred_recurrent_recon': pred_masked_recons,
                     'true_recurrent_recon': true_masked_recons[1:],      # exclude first, as this is the zero vector
                     'hidden': hidden_embedding
                     }

        return partly_recon.reshape(x.shape), meta_data

    def loss_function(self, x, x_recon, meta) -> dict:
        x = meta["true_recurrent_recon"][-1]

        x_flat = torch.flatten(x, start_dim=1)
        recon_flat = torch.flatten(x_recon, start_dim=1)
        recons_loss = F.mse_loss(x_flat, recon_flat)

        true_bank = meta['true_bank']
        bank_loss = 0
        for level in range(len(true_bank)):
            bank_loss += F.mse_loss(true_bank[level], meta["filter_bank"][level])
        return {'loss': recons_loss, 'bank_loss': bank_loss}

    def training_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        recon, meta_results = self(sequences, teacher_forcing=self.teacher_forcing)
        train_loss_dict = self.loss_function(sequences, recon, meta=meta_results)
        self.log("train_loss", train_loss_dict['loss'], prog_bar=True, logger=True)
        return train_loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        recon, meta_results = self(sequences, teacher_forcing=0.)
        val_loss_dict = self.loss_function(sequences, recon, meta=meta_results)
        self.log("val_loss", val_loss_dict['loss'], prog_bar=True, logger=True)
        return val_loss_dict['loss']

    def test_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        recon, meta_results = self(sequences, teacher_forcing=0.)
        test_loss_dict = self.loss_function(sequences, recon, meta=meta_results)
        self.log("test_loss", test_loss_dict['loss'], prog_bar=True, logger=True)
        return test_loss_dict['loss']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters()) #, lr=0.01)
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


def create_vec2seq(recurrent_net,
                   wavelet='haar',
                   auto_reccurent=False,
                   teacher_forcing_ratio=0.5,
                   coarse_skip=0,
                   recursion_limit=None,
                   dir_path="configs/logs", verbose=False, monitor="val_loss", mode="min",
                   num_epochs=20, gpus=1,
                   validation_hook_batch=None, test_hook_batch=None, project='WaveLSTM', run_id="null"):

    _model = Vec2Seq(recurrent_net=recurrent_net,
                     wavelet=wavelet,
                     auto_reccurent=auto_reccurent,
                     teacher_forcing_ratio=teacher_forcing_ratio,
                     coarse_skip=coarse_skip,
                     recursion_limit=recursion_limit,
                     )
    print(_model)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project=project,
                               name=run_id,
                               job_type='train')

    # Make all callbacks
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
        min_delta=0.02,
        patience=2,
        verbose=verbose
    )

    viz_embedding_callback = LatentSpace(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    viz_prediction_callback = FeatureSpace1d(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    viz_rnn_callback = RecurrentFeatureSpace1d(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    callbacks = [checkpoint_callback,
                 early_stop_callback,
                 viz_embedding_callback,
                 viz_prediction_callback,
                 viz_rnn_callback
                 ]

    _trainer = pl.Trainer(
        default_root_dir=dir_path,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        max_epochs=num_epochs,
        check_val_every_n_epoch=5,  # int(np.floor(num_epochs/3)),
        gpus=gpus,
    )

    return _model, _trainer
