from abc import ABC
# Torch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
# PyTorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger              # tracking tool
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import ptwt
import pywt
# Modules
from src.WaveLSTM.models.base import WaveletBase as SourceSeparation
from src.WaveTransformer.modules.Conv1dTransformer import ConvTransformer
# Callbacks
from src.WaveTransformer.custom_callbacks import waveTransformer
# from WaveTransformer.custom_callbacks import attention
# from WaveTransformer.custom_callbacks import autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentiveAutoEncoder(pl.LightningModule, ABC):

    def get_conv_shape(self, width, kernel_size, padding=0, stride=1):
        return int( (( width - kernel_size + (2 * padding)) / stride ) + 1 )

    def __init__(self, config):
        """
        """
        super().__init__()
        self.save_hyperparameters()

        # Data
        self.input_channels = config.encoder.cnn.in_channel
        self.output_channels = config.encoder.cnn.out_channel
        self.nhead = config.encoder.transformer.nhead
        self.hdim = config.encoder.transformer.dim
        self.length = config.encoder.base.input_length
        self.input_size = self.length
        self.pool_targets = config.experiment.pool_targets     # Reconstruct in pooled, or original feature space

        # Encoder
        self.source_separation_layer = SourceSeparation(input_size=self.length,
                                                        input_channels=self.input_channels,
                                                        recursion_limit=config.encoder.wave.J,
                                                        wavelet=config.encoder.wave.wavelet)
        self.encoder = ConvTransformer(in_channel=self.input_channels,
                                        out_channel = self.output_channels,
                                        length = self.length,
                                        dim=self.hdim,
                                        nhead = self.nhead,
                                        J = config.encoder.wave.J)
        # Decoder:
        decoder_method=  config.decoder.base.method.lower()
        self.target_width = self.source_separation_layer.masked_width if self.pool_targets else self.input_size
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
                nn.Linear(config.encoder.cnn.out_channel * self.hdim * config.encoder.wave.J, 32 * w_cnn),
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
        elif decoder_method == "linear":
            w_cnn = self.target_width * self.input_channels
            self.decoder = nn.Sequential(
                torch.nn.Flatten(start_dim=1),
                nn.Linear(config.encoder.cnn.out_channel * self.hdim * config.encoder.wave.J, w_cnn),
                # nn.ReLU(),
                # nn.Conv1d(256, self.input_channels, 1, stride=1),
                nn.Flatten(start_dim=1),
                nn.Unflatten(1, (self.input_channels, self.target_width))
            )
        else:
            raise NotImplementedError

    def forward(self, x: torch.tensor):
        """
        """
        assert x.dim() == 3

        # print(x.shape)
        # print(x[0][0])

        # Input masking. Perform IWT twice
        # - first for if we want to reconstruct with filterered targets (see loss: default False)
        masked_inputs, masked_targets = self.source_separation_layer(x, pool_targets=self.pool_targets)
        # - second for the input sequence for the waveTransformer encoder
        meta_data = {
            'masked_inputs': [m_i.detach().cpu().numpy() for m_i in masked_inputs],
            'masked_targets': [m_t.detach().cpu().numpy() for m_t in masked_targets],
            'masked_targets_tensor': masked_targets,
        }

        # Attentively encode
        M, res_embeds, attention= self.encoder(masked_inputs)
        meta_data.update({"M": M.detach().cpu().numpy()})
        meta_data.update({"Attn": attention[0].detach().cpu().numpy()})
        meta_data.update({"resolution_embeddings": [_hidden.detach().cpu().numpy() for _hidden in res_embeds]})                             # [batch_size, attention-hops, resolution_embed_size]

        # Decode
        recon = self.decoder(M)                                                      # [batch_size, channels, width]
        meta_data.update({'masked_predictions': [recon.detach().cpu().numpy()],
                          })
        
        print("reconstruction", recon.shape)

        return recon, meta_data

    def loss(self, batch, batch_idx, filter=False) -> dict:
        recon, meta_results = self(batch['CNA'])
        target = meta_results["masked_targets_tensor"][-1] if filter else batch['CNA']
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
        optimizer = optim.AdamW(self.parameters())
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
    
def stack_batches(dataloader):
    # Stack all test data for test hook
    CNA, labels = [], []
    for batch in iter(dataloader):
        CNA.append(batch["CNA"])
        labels.append(batch["label"])
    return {"CNA": torch.concat(CNA, 0),
            "label": torch.concat(labels, 0),}

def create_sa_autoencoder(data_module, cfg, gpus=1):

    # Get validation and test hook batch
    val_data = next(iter(data_module.val_dataloader()))
    test_data = stack_batches(data_module.test_dataloader())

    _model = AttentiveAutoEncoder(cfg)
    logging.debug(_model)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project=cfg.experiment.project_name,
                               name=cfg.experiment.run_id,
                               job_type='train',
                               save_dir="outputs")

    # Make all callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        filename=cfg.experiment.run_id,
        verbose=cfg.experiment.verbose,
        monitor="val_loss",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", mode="min",
        min_delta=0.00,
        patience=10,
        verbose=cfg.experiment.verbose,
    )

    viz_res_embedding = waveTransformer.ResolutionEmbedding(
        val_samples=val_data,
        test_samples=test_data
    )

    # viz_multi_res_embed = attention.MultiResolutionEmbedding(
    #     val_samples=val_data,
    #     test_samples=test_data
    # )

    # viz_attention = attention.Attention(
    #     val_samples=val_data,
    #     test_samples=test_data
    # )

    # viz_reconstruction = autoencoder.Reconstruction(
    #     val_samples=val_data,
    #     test_samples=test_data
    # )

    save_output = waveTransformer.SaveOutput(
        test_samples=test_data,
        file_path=cfg.experiment.save_file
    )


    callbacks = [checkpoint_callback,
                 early_stop_callback,
                 viz_res_embedding,
                #  viz_multi_res_embed,
                #  viz_attention,
                #  viz_reconstruction,
                 save_output,
                 ]

    _trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=cfg.experiment.num_epochs,
        check_val_every_n_epoch=5,
        log_every_n_steps=2,
        accelerator='gpu',
        strategy="ddp",
        devices=[0],
    )

    return _model, _trainer