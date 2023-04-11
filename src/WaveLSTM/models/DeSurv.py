from abc import ABC
# Torch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
# PyTorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger  # tracking tool
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import ptwt
import pywt

from WaveLSTM.modules.self_attentive_encoder import SelfAttentiveEncoder
from WaveLSTM.custom_callbacks import waveLSTM_callbacks
from WaveLSTM.custom_callbacks import ae_callbacks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentiveDeSurv(pl.LightningModule, ABC):

    def __init__(self,
                 encoder_config,
                 config, ):
        super().__init__()

        raise NotImplemented

        # Encoder
        self.self_attentive_encoder = SelfAttentiveEncoder(encoder_config=encoder_config,
                                                           config=config)

        # DeSurv network
        # self.clf_net = nn.Sequential(nn.LazyLinear(config['nfc']),
        #                              nn.ReLU(),
        #                              nn.Linear(config['nfc'], len(config['classes'])))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.tensor):
        m, meta_results = self.self_attentive_encoder(x)

        output = output.view(m.size(0), -1)  # [batch_size, attention-hops * n_hidden]
        pred = self.clf_net(output)

        meta_results.update({"M": m})

        return pred, meta_results

    def loss_function(self, targets, pred_t, meta) -> dict:
        bsz = meta["attention"].size(0)  # batch size

        # Classifier loss
        clf_loss = self.criterion(pred_t, targets)

        # Calculate penalization to ensure diversity in hops
        atn_t = torch.transpose(meta["attention"], 1, 2).contiguous()
        eye = torch.stack([torch.eye(self.attention_hops) for _ in range(bsz)], dim=0)
        p = torch.norm(torch.bmm(meta["attention"], atn_t))  # - eye.to(device))
        pen_loss = clf_loss + self.attention_pen * p

        # accuracy
        _, pred_label = torch.max(pred_t, dim=1)
        acc = torch.sum(pred_label == targets).item() / bsz

        return {'loss': pen_loss,
                'clf_loss': clf_loss,
                'acc': acc}

    def training_step(self, batch, batch_idx):
        sequences, targets = batch['feature'], batch['label']
        pred_t, meta_results = self(sequences)
        train_loss_dict = self.loss_function(targets, pred_t, meta=meta_results)
        self.log("train_loss", train_loss_dict['loss'], prog_bar=True, logger=True)
        self.log("train_clf_loss", train_loss_dict['clf_loss'], prog_bar=True, logger=True)
        self.log("train_acc", train_loss_dict['acc'], prog_bar=True, logger=True)
        return train_loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        sequences, targets = batch['feature'], batch['label']
        pred_t, meta_results = self(sequences)
        val_loss_dict = self.loss_function(targets, pred_t, meta=meta_results)
        self.log("val_loss", val_loss_dict['loss'], prog_bar=True, logger=True)
        self.log("val_clf_loss", val_loss_dict['clf_loss'], prog_bar=True, logger=True)
        self.log("val_acc", val_loss_dict['acc'], prog_bar=True, logger=True)
        return val_loss_dict['loss']

    def test_step(self, batch, batch_idx):
        sequences, targets = batch['feature'], batch['label']
        pred_t, meta_results = self(sequences)
        test_loss_dict = self.loss_function(targets, pred_t, meta=meta_results)
        self.log("test_loss", test_loss_dict['loss'], prog_bar=True, logger=True)
        self.log("test_clf_loss", test_loss_dict['clf_loss'], prog_bar=True, logger=True)
        self.log("test_acc", test_loss_dict['acc'], prog_bar=True, logger=True)
        return test_loss_dict['loss']

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer),  # The scheduler instance
            "interval": "epoch",  # The unit of the scheduler's step size
            "frequency": 5,  # How many epochs/steps should pass between calls to `scheduler.step()`
            "monitor": "val_loss",  # Metric to monitor for scheduler
            "strict": True,  # Enforce that "val_loss" is available when the scheduler is updated
            "name": 'LearningRateMonitor',  # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }


def create_single_risk(classes, seq_length, strands, chromosomes,
                       hidden_size=256, layers=1, proj_size=0,
                       wavelet='haar',
                       recursion_limit=None,
                       dir_path="configs/logs", verbose=False, monitor="val_loss", mode="min",
                       num_epochs=20, gpus=1,
                       validation_hook_batch=None, test_hook_batch=None,
                       project='WaveLSTM-classifier', run_id="null"):
    encoder_config = {"seq_length": seq_length,
                      "strands": strands,
                      "chromosomes": chromosomes,
                      "hidden_size": hidden_size,
                      "layers": layers,
                      "proj_size": proj_size,
                      "wavelet": wavelet,
                      "recursion_limit": recursion_limit
                      }
    config = {"dropout": 0.,
              "attention-unit": 350,
              "attention-hops": 10,
              "penalization_coeff": 0.,
              "nfc": 128,  # hidden layer size for MLP in the classifier
              "classes": classes,  # number of class for the last step of classification
              "real_hidden_size": proj_size if proj_size > 0 else hidden_size,  # Encoder's hidden size
              }

    label_dictionary = {key: val for key, val in zip([i for i in range(len(classes))], classes)}

    _model = SelfAttentiveEncoder(encoder_config=encoder_config, config=config)
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
        min_delta=0.0,
        patience=10,
        verbose=verbose
    )

    viz_embedding_callback = waveLSTM_callbacks.ViewEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch,
        label_dictionary=label_dictionary
    )

    viz_attention_1 = clf_callbacks.ClfMultiScaleEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch,
        label_dictionary=label_dictionary
    )

    save_output = waveLSTM_callbacks.SaveOutput(
        test_samples=test_hook_batch,
    )

    callbacks = [checkpoint_callback,
                 early_stop_callback,
                 viz_embedding_callback,
                 viz_attention_1,
                 save_output
                 ]

    _trainer = pl.Trainer(
        default_root_dir=dir_path,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        max_epochs=num_epochs,
        check_val_every_n_epoch=2,
        gpus=gpus,
        log_every_n_steps=5
    )

    return _model, _trainer
