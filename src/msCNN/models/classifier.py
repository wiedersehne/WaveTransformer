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
# Modules
from src.msCNN.modules.mscnn import PREDICTION_HEAD, MSCNN_NET 
from src.msCNN.modules.self_attentive_encoder import SelfAttentiveEncoder
from src.msCNN.custom_callbacks import attention
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentiveClassifier(pl.LightningModule, ABC):

    def __init__(self, config):

        super().__init__()
        self.save_hyperparameters()

        # Data
        self.seq_length = config.encoder.base.input_length

        self.linear = nn.Conv1d(config.encoder.base.input_size, config.encoder.cnns.out_channel,  1, stride=1)

        self.encoder = MSCNN_NET(
                                config.encoder.cnns.in_channel,
                                config.encoder.cnns.out_channel,
                                config.encoder.cnns.kernels,
                                config.encoder.cnns.layers
        )

        # Classifier network
        self.clf_net = PREDICTION_HEAD(
                                self.seq_length*config.encoder.cnns.out_channel,
                                config.classifier.hidden_dimension1,
                                config.classifier.hidden_dimension2,
                                config.classifier.num_classes
                                )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.tensor, **kwargs):

        assert x.dim() == 3

        meta_data = {}
        
        output, res_embeds = self.encoder(self.linear(x))
        meta_data.update({"resolution_embeddings": [_hidden.detach().cpu().numpy() for _hidden in res_embeds]})
        output = output.view(output.size(0), -1)

        # Decode
        pred, M = self.clf_net(output)
        meta_data.update({"M": M.detach().cpu().numpy()})

        return pred, meta_data

    def loss_function(self, targets, pred_t, meta) -> dict:

        bsz = targets.shape[0]                   # batch size

        # Classifier loss
        clf_loss = self.criterion(pred_t, targets)
        
        # atn = meta["attention"]
        # Calculate penalization (promotes diversity in hops)
        # atn_t = torch.transpose(atn, 1, 2).contiguous()
        # eye = torch.stack([torch.eye(self.encoder.attention_hops) for _ in range(bsz)], dim=0)
        # p = torch.norm(torch.bmm(atn, atn_t) - eye.to(device))
        # penalty = self.a_encoder.attention_pen * p

        # Record classification accuracy
        _, pred_label = torch.max(pred_t, dim=1)
        acc = torch.sum(pred_label == targets).item() / bsz

        return {'loss': clf_loss, #+ penalty,              # Penalized loss
                'acc': acc}                              # Accuracy

    def training_step(self, batch, batch_idx):
        sequences, targets = batch['CNA'], batch['label']
        pred_t, meta_results = self(sequences)
        train_loss_dict = self.loss_function(targets, pred_t, meta=meta_results)
        self.log("train_loss", train_loss_dict['loss'], prog_bar=True, logger=True)
        self.log("train_acc", train_loss_dict['acc'], prog_bar=True, logger=True)
        return train_loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        sequences, targets = batch['CNA'], batch['label']
        pred_t, meta_results = self(sequences)
        val_loss_dict = self.loss_function(targets, pred_t, meta=meta_results)
        self.log("val_loss", val_loss_dict['loss'], prog_bar=True, logger=True)
        self.log("val_acc", val_loss_dict['acc'], prog_bar=True, logger=True)
        return val_loss_dict['loss']

    def test_step(self, batch, batch_idx):
        sequences, targets = batch['CNA'], batch['label']
        pred_t, meta_results = self(sequences)
        test_loss_dict = self.loss_function(targets, pred_t, meta=meta_results)
        self.log("test_loss", test_loss_dict['loss'], prog_bar=True, logger=True)
        self.log("test_acc", test_loss_dict['acc'], prog_bar=True, logger=True)
        return test_loss_dict['loss']

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
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


def create_classifier(classes, data_module, cfg, gpus=1):

    # Get validation and test hook batch
    val_data = next(iter(data_module.val_dataloader()))
    test_data = next(iter(data_module.test_dataloader()))

    _model = AttentiveClassifier(config=cfg)
    logging.debug(_model)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project=cfg.experiment.project_name,
                               name=cfg.experiment.run_id,
                               job_type='train',
                               save_dir=cfg.experiment.output_dir
                               )

    # Make all callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.experiment.output_dir}checkpoints",
        filename=cfg.experiment.run_id,
        verbose=cfg.experiment.verbose,
        monitor="val_loss",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", mode="min",
        min_delta=0,
        patience=10,
        verbose=cfg.experiment.verbose,
    )

    # TODO: add this to the data module to avoid bugs getting order wrong
    label_dictionary = {key: val for key, val in zip([i for i in range(len(classes))], classes)}

    # viz_embedding_callback = waveLSTM.ResolutionEmbedding(
    #     val_samples=val_data,
    #     test_samples=test_data,
    #     label_dictionary=label_dictionary
    # )

    viz_multi_res_embed = attention.MultiResolutionEmbedding(
        val_samples=val_data,
        test_samples=test_data,
        label_dictionary=label_dictionary
    )

    # viz_attention = attention.Attention(
    #     val_samples=val_data,
    #     test_samples=test_data
    # )

    # save_output = waveLSTM.SaveOutput(
    #     test_samples=test_data,
    #     file_path=cfg.experiment.save_file
    # )

    callbacks = [checkpoint_callback,
                 early_stop_callback,
                #  viz_embedding_callback,
                 viz_multi_res_embed,
                #  viz_attention,
                #  save_output
                 ]

    _trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=cfg.experiment.num_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=2,
        devices=gpus,
    )

    return _model, _trainer
