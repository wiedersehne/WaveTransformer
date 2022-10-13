from abc import ABC
# Torch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# PyTorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger              # tracking tool
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# Torchmetrics
from torchmetrics.functional import accuracy
# Local
from source.custom_callbacks.classifier_callbacks import SequencePrediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(pl.LightningModule, ABC):

    def __init__(self, model):
        """
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(path)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def loss(logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        sequences, filter_bank = batch['feature']
        labels = batch['label']
        logits = torch.log_softmax(self(sequences), dim=1)
        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", acc, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": acc}

    def validation_step(self, batch, batch_idx):
        sequences, filter_bank = batch['feature']
        labels = batch['label']
        logits = torch.log_softmax(self(sequences), dim=1)
        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": acc}

    def test_step(self, batch, batch_idx):
        sequences, filter_bank = batch['feature']
        labels = batch['label']
        logits = torch.log_softmax(self(sequences), dim=1)
        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", acc, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": acc}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer),         # The scheduler instance
            "interval": "epoch",                               # The unit of the scheduler's step size
            "frequency": 1,                     # How many epochs/steps should pass between calls to `scheduler.step()`
            "monitor": "val_loss",              # Metric to monitor for scheduler
            "strict": True,                     # Enforce that "val_loss" is available when the scheduler is updated
            "name": 'LearningRateMonitor',      # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }


def create_classifier(model,
                      dir_path="configs/logs", verbose=False, monitor="val_loss", mode="min",
                      num_epochs=100, gpus=1,
                      validation_hook_batch=None):
    """ Decorated classifier model wrapper """

    _model = Classifier(model=model)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project='Classifier', job_type='train')

    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_path + "/checkpoints",
        filename='best_checkpoint',
        save_top_k=1,
        verbose=verbose,
        monitor=monitor,
        mode=mode,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_accuracy", mode="max",
        patience=10,
        verbose=verbose
    )

    callbacks = [checkpoint_callback, early_stop_callback]
    if validation_hook_batch is not None:
        callbacks.append(SequencePrediction(validation_hook_batch))

    _trainer = pl.Trainer(
        default_root_dir=dir_path,
        logger=wandb_logger,
        callbacks=callbacks,
        checkpoint_callback=True,
        max_epochs=num_epochs,
        check_val_every_n_epoch=1,
        gpus=gpus,
    )

    return _model, _trainer
