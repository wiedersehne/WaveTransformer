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
from WaveLSTM.models.base import WaveletBase
from WaveLSTM.modules.self_attentive_encoder import SelfAttentiveEncoder
# Callbacks
from WaveLSTM.custom_callbacks import waveLSTM
from WaveLSTM.custom_callbacks import attention
from WaveLSTM.custom_callbacks import autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentiveAutoEncoder(pl.LightningModule, ABC, WaveletBase):

    def get_conv_shape(self, width, kernel_size, padding=0, stride=1):
        return int( (( width - kernel_size + (2 * padding)) / stride ) + 1 )

    @property
    def encoder(self):
        return self.a_encoder.encoder

    def __init__(self,
                 encoder_config,
                 attention_config,
                 decoder_config,
                 ):
        super().__init__()
        WaveletBase.__init__(self,
                             seq_length=encoder_config["seq_length"],
                             recursion_limit=encoder_config["recursion_limit"],
                             wavelet=encoder_config["wavelet"])

        self.save_hyperparameters()
        self.channels, self.seq_length = encoder_config["channels"], encoder_config["seq_length"]

        # Encoder
        encoder_config["J"] = self.J
        encoder_config["pooled_width"] = self.masked_width
        self.a_encoder = SelfAttentiveEncoder(encoder_config=encoder_config,
                                              config=attention_config)

        # Decoder:

        # Convolutional:  Decode from the multi-resolution embedding (M) using a transposed convolutional neural network
        # Recursively work backwards, to find what the linear layer's output dim should be
        # ... a linear layer is needed as we aren't just tranposing (inverting) a CNN encoder
        # ... ... i.e. we don't automatically encode to a correct pooled width.
        if decoder_config["arch"] == "cnn":
            k_size = 7
            stride = 1
            # w_cnn = self.seq_length
            # for cnn_layer in range(4):
            #     w_cnn = self.get_conv_shape(w_cnn, k_size, stride=stride)
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(attention_config["attention-hops"], 8, k_size, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(8, 16, k_size, stride=stride),
                nn.ReLU(),
                nn.ConvTranspose1d(16, 64, k_size, stride=stride),
                nn.ReLU(),
                nn.ConvTranspose1d(64, self.encoder.channels, k_size, stride=stride),
                nn.ReLU(),
                torch.nn.Flatten(start_dim=1),
                nn.LazyLinear(self.encoder.channels * self.seq_length),
                torch.nn.Unflatten(1, (self.encoder.channels, self.seq_length))
            )

        elif decoder_config["arch"] == "fc":
            # Fully connected
            nfc = decoder_config["nfc"]
            w = self.seq_length
            self.decoder = nn.Sequential(
                torch.nn.Flatten(start_dim=1),
                nn.Linear(encoder_config["scale_embed_dim"] * attention_config["attention-hops"], nfc),
                nn.ReLU(),
                nn.Linear(nfc, w * self.encoder.channels),
                nn.Unflatten(1, (self.encoder.channels, w)),
            )

    def forward(self, x: torch.tensor):
        """
        """
        assert x.dim() == 3

        # Input masking
        _, masked_targets = self.sequence_mask(x)

        scaled_masked_inputs, _ = self.sequence_mask(self.scale(x))
        meta_data = {
            'masked_inputs': scaled_masked_inputs,
            'masked_targets': masked_targets,
        }

        # Attentively encode
        output, meta_data = self.a_encoder(scaled_masked_inputs, meta_data)
        meta_data.update({"M": output})                                 # [batch_size, attention-hops, scale_embed_dim]
        # output = output.view(output.size(0), -1)                      # [batch_size, attention-hops * scale_embed_dim]

        # Decode
        recon = self.decoder(output)                                    # [batch_size, channels, width]
        meta_data.update({'masked_predictions': [recon],
                          })

        return recon, meta_data

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

def create_sa_autoencoder(seq_length, channels,
                          hidden_size=256, layers=1, proj_size=0, scale_embed_dim=128,
                          r_hops=1, decoder="fc", nfc=256,
                          wavelet='haar',
                          recursion_limit=None,
                          dir_path="logs", verbose=False,  monitor="val_loss",
                          num_epochs=20, gpus=1,
                          validation_hook_batch=None, test_hook_batch=None,
                          project='WaveLSTM-attentive-autoencoder', run_id="null",
                          outfile="logs/a-ae-output.pkl",
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
                        "attention-hops": r_hops,
                        "real_hidden_size": proj_size if proj_size > 0 else hidden_size,       # Encoder's hidden size
                        }
    decoder_config = {"arch": decoder,
                      "nfc": nfc}

    _model = AttentiveAutoEncoder(encoder_config=encoder_config,
                                  attention_config=attention_config,
                                  decoder_config=decoder_config,
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
        min_delta=0.00,
        patience=10,
        verbose=verbose
    )

    viz_res_embedding = waveLSTM.ResolutionEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    viz_multi_res_embed = attention.MultiResolutionEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    viz_attention = attention.Attention(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    viz_reconstruction = autoencoder.Reconstruction(
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
                 viz_multi_res_embed,
                 viz_attention,
                 viz_reconstruction,
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
