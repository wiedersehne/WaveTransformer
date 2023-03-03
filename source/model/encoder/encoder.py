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
from source.custom_callbacks.callback_waveLSTM import *
from source.model.encoder.WaveConvLSTM import WaveletConv1dLSTM

import ptwt
import pywt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(pl.LightningModule, ABC):

    def fwt(self, x):
        # Fast wavelet transform, splitting along the strand index, stacking over chromosomes
        filter_bank = ptwt.wavedec(x.reshape((x.size(0), -1)), self.wavelet, mode='zero', level=self.max_level)
        return filter_bank

    def masked_truth(self, x):
        """
        Get the target sequence, from sequentially reconstructing masked (finer scales), true coefficients
        """
        true_bank = self.fwt(x)
        masked_truth = [torch.zeros_like(x, device=device)]
        for t in range(self.recursion_limit):
            masked_bank = [coeff if i <= t else torch.zeros_like(coeff) for i, coeff in enumerate(true_bank)]
            next_recon = ptwt.waverec(masked_bank, self.wavelet).reshape(x.shape)
            masked_truth.append(next_recon)
        return masked_truth

    def __init__(self,
                 seq_length, strands, chromosomes,
                 hidden_size=256, layers=1, proj_size=0,
                 wavelet="haar",
                 coarse_skip: int = 0,
                 recursion_limit=None,
                 ):

        assert coarse_skip >= 0

        super().__init__()
        self.save_hyperparameters()

        l = seq_length * strands * chromosomes

        # Wavelet
        self.wavelet = pywt.Wavelet(wavelet)
        # # Calculate the filter bank sizes, and recursion depth
        self.max_level = pywt.dwt_max_level(l, wavelet) - coarse_skip
        self.full_bank_lengths = [t.shape[1] for t in pywt.wavedec(np.zeros((1, l)), self.wavelet, level=self.max_level)]
        # Detail space lengths, then insert first approximation space length
        self.recursion_limit = len(self.full_bank_lengths) if recursion_limit is None \
            else np.min((recursion_limit, len(self.full_bank_lengths)))
        self.bank_lengths = self.full_bank_lengths[:recursion_limit]

        # Encoder/rec_net
        self.rec_net = WaveletConv1dLSTM(out_features=seq_length, strands=strands, chromosomes=chromosomes,
                                         bank_lengths=self.bank_lengths,
                                         hidden_size=hidden_size, layers=layers, proj_size=proj_size)

    def __str__(self):
        s = ''
        s += f'\nData'
        s += f'\n\t Data size (B, {self.rec_net.C}, {self.rec_net.H}, {self.rec_net.W})'
        s += f'\nRecurrent network'
        s += f'\n\t Wavelet "{self.wavelet.name}", which has decomposition length {self.wavelet.dec_len}'
        s += f'\n\t Full filter bank lengths {self.full_bank_lengths}'
        s += f'\n\t Recursion is over bank lengths {self.bank_lengths}, with recursion limit {self.recursion_limit}'
        s += str(self.rec_net)
        return s

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(path)

    def forward(self, x: torch.tensor):
        """
        x,                 features = [B, Strands, Chromosomes, Sequence Length]
        """
        if x.dim() == 3:
            x = x.unsqueeze(2)

        # Partial fidelity truth, removing (zeroing) the contribution of coefficients further down the recurrence
        masked_truths = self.masked_truth(x)

        # Recurrent outputs:
        #  the approximation/detail space coefficients, which gets refined at each step
        #  the corresponding reconstructed features
        #  the hidden embedding at that scale
        predicted_bank = [torch.zeros((x.size(0), length), device=device) for length in self.full_bank_lengths]
        pred_masked_recons = []

        # Initialise hidden and cell states, and first LSTM input (usually this would be a <SOS> token in NLP)
        hidden_state = self.rec_net.init_states(x)
        hidden_embedding = []
        partly_recon = torch.zeros(x.shape, device=device)
        for t in range(self.recursion_limit):

            predicted_bank[t], hidden_state, latent = self.rec_net(x - masked_truths[t], hidden_state, t)
            partly_recon = ptwt.waverec(predicted_bank, self.wavelet).reshape(x.shape)

            # Record next output for target sequence
            pred_masked_recons.append(partly_recon)
            hidden_embedding.append(latent)

        meta_data = {'filter_bank': predicted_bank,
                     'pred_recurrent_recon': pred_masked_recons,
                     'true_recurrent_recon': masked_truths[1:],      # exclude first, as this is the zero vector
                     'hidden': hidden_embedding
                     }

        return partly_recon.reshape(x.shape), meta_data

    def loss_function(self, x, x_recon, meta) -> dict:
        x = meta["true_recurrent_recon"][-1]

        x_flat = torch.flatten(x, start_dim=1)
        recon_flat = torch.flatten(x_recon, start_dim=1)
        recons_loss = F.mse_loss(x_flat, recon_flat)

        # true_bank = meta['true_bank']
        # bank_loss = 0
        # for level in range(len(true_bank)):
        #     bank_loss += F.mse_loss(true_bank[level], meta["filter_bank"][level])
        return {'loss': recons_loss}    # , 'bank_loss': 0

    def training_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        recon, meta_results = self(sequences)
        train_loss_dict = self.loss_function(sequences, recon, meta=meta_results)
        self.log("train_loss", train_loss_dict['loss'], prog_bar=True, logger=True)
        return train_loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        recon, meta_results = self(sequences)
        val_loss_dict = self.loss_function(sequences, recon, meta=meta_results)
        self.log("val_loss", val_loss_dict['loss'], prog_bar=True, logger=True)
        return val_loss_dict['loss']

    def test_step(self, batch, batch_idx):
        sequences, labels = batch['feature'], batch['label']
        recon, meta_results = self(sequences)
        test_loss_dict = self.loss_function(sequences, recon, meta=meta_results)
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
                       hidden_size=256, layers=1, proj_size=0,
                       wavelet='haar',
                       coarse_skip=0,
                       recursion_limit=None,
                       dir_path="configs/logs", verbose=False, monitor="val_loss", mode="min",
                       num_epochs=20, gpus=1,
                       validation_hook_batch=None, test_hook_batch=None, project='WaveLSTM', run_id="null"):

    _model = Encoder(seq_length, strands, chromosomes,
                     hidden_size=hidden_size, layers=layers, proj_size=proj_size,
                     wavelet=wavelet,
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

    viz_embedding_callback = ViewEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    viz_prediction_callback = ViewSignal(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    viz_rnn_callback = ViewRecurrentSignal(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    callbacks = [checkpoint_callback,
                 early_stop_callback,
                 viz_prediction_callback,
                 viz_rnn_callback,
                 viz_embedding_callback
                 ]

    _trainer = pl.Trainer(
        default_root_dir=dir_path,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        max_epochs=num_epochs,
        check_val_every_n_epoch=5,
        gpus=gpus,
    )

    return _model, _trainer
