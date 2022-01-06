from abc import ABC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import accuracy
from experiments.configs.config import extern
from .embedding_module import EmbeddingModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceModel(nn.Module):

    def __init__(self, n_classes: int,
                 n_hidden: int = 256,
                 n_layers: int = 3,
                 dropout: float = 0.6,
                 bidirectional: bool = False,
                 stack: int = 1,
                 out_channels: int = 23,
                 kernel_size: int = 3,
                 stride: int = 5,
                 padding: int = 1):
        """

        :param n_classes:
        :param n_hidden:
        :param n_layers:
        :param dropout:
        :param bidirectional:
        :param stack:                           Dimension to stack chromosomes on
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        """
        super().__init__()
        self.hidden_size = n_hidden
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.stack_dim = stack                          # 1 => Major/min separated by feature. 2=> sequences appended

        self.conv1d_major = nn.Conv1d(
            in_channels=23,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv1d_minor = nn.Conv1d(
            in_channels=23,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.lstm = nn.LSTM(
            input_size=int((250/1)*self.stack_dim),
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        if bidirectional is False:
            self.fc = nn.Linear(n_hidden, n_classes)
        else:
            self.fc = nn.Linear(2 * n_hidden, n_classes)

    def forward(self, x):

        # Split sequences and re-order for CNN layer
        x_major = torch.transpose(x[:, :, :, 0], 1, 2)
        x_minor = torch.transpose(x[:, :, :, 1], 1, 2)

        # Independent CNN networks for each sequence
        x_major = self.conv1d_major(x_major)                   # chromosomes are channels of CNN,
        x_minor = self.conv1d_minor(x_minor)                   # and length gets down-sampled from striding

        # Concatenate either along sequence (dim=2) or feature dimension (dim=1)
        x = torch.cat([x_major, x_minor], dim=self.stack_dim)  # Stack output channels of CNN as features to LSTM

        # Test skipping LSTM part
        # return self.fc(x.reshape((x.shape[0], -1)))

        # LSTM layers
        self.lstm.flatten_parameters()                         # For multiple GPU cases
        if self.bidirectional is False:
            _, (hidden, _) = self.lstm(x)
            out = hidden[-1]
        else:
            h0 = torch.zeros(self.n_layers * 2, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.n_layers * 2, x.size(0), self.hidden_size).to(device)
            out, (hn, cn) = self.lstm(x, (h0, c0))
            out = out[:, -1, :]

        # return through fully connected layer
        return self.fc(out)


class PredictorCN(pl.LightningModule, ABC, EmbeddingModule):

    @staticmethod
    def batch_to_data(batch):
        # TODO: This method should be held in pipeline
        # From batch, get model input and output
        x = torch.stack((batch["nmajor"], batch["nminor"]), dim=-1).squeeze(2)
        y = batch["cancer_type"].squeeze(1).long()
        return x, y

    def __init__(self, n_classes: int,
                 n_hidden: int = 256,
                 n_layers: int = 3,
                 dropout: float = 0,
                 bidirectional: bool = True,
                 stack: int = 1,
                 out_channels: int = 23,
                 kernel_size: int = 3,
                 stride: int = 5,
                 padding: int = 1,
                 ):
        """

        :param n_classes:
        :param n_hidden:
        :param n_layers:
        :param dropout:
        :param bidirectional:
        :param stack:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        """

        super().__init__()
        self.n_classes = n_classes
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.stack = stack
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.model = SequenceModel(n_classes, n_hidden=n_hidden, n_layers=n_layers, dropout=dropout,
                                   bidirectional=bidirectional, stack=stack, out_channels=out_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding)
        self.criterion = nn.CrossEntropyLoss()

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(path,
                                         n_classes=self.n_classes, n_hidden=self.n_hidden, n_layers=self.n_layers,
                                         dropout=self.dropout, bidirectional=self.bidirectional, stack=self.stack,
                                         out_channels=self.out_channels, kernel_size=self.kernel_size,
                                         stride=self.stride, padding=self.padding,
                                         )

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)

        return loss, output

    def training_step(self, batch, batch_idx):
        sequences, labels = self.batch_to_data(batch)
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_step(self, batch, batch_idx):
        sequences, labels = self.batch_to_data(batch)
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def test_step(self, batch, batch_idx):
        sequences, labels = self.batch_to_data(batch)
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", step_accuracy, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_accuracy}

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


@extern
def model_constructor(n_classes, bidirectional=True, lstm_dropout=0., dir_path="experiments/logs", verbose=False,
                      monitor="val_loss", mode="min", num_epochs=100, gpus=1, pb_refresh=10, n_hidden=256,
                      n_layers=3, stack=1, out_channels=23, kernel_size=3, stride=5, padding=1):
    """Wrapper. Mostly for .yaml configuration and readability purposes """
    model_ = PredictorCN(
        n_classes=n_classes,
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout=lstm_dropout,
        bidirectional=bidirectional,
        stack=stack,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_path + "/checkpoints",
        filename='best_checkpoint',
        save_top_k=1,
        verbose=verbose,
        monitor=monitor,
        mode=mode,
    )

    trainer_ = pl.Trainer(
        default_root_dir=dir_path,
        callbacks=checkpoint_callback,
        checkpoint_callback=True,
        max_epochs=num_epochs,
        gpus=gpus,
        progress_bar_refresh_rate=pb_refresh,
    )

    return model_, trainer_
