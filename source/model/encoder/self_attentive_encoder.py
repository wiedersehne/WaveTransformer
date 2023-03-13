from abc import ABC
# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# PyTorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger              # tracking tool
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from source.custom_callbacks.waveLSTM_callbacks import *
from source.custom_callbacks.atn_callbacks import *
from source.model.encoder.encoder import Encoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelfAttentiveEncoder(pl.LightningModule, ABC):

    def __init__(self,
                 encoder_config,
                 config,
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.convlstm = Encoder(**encoder_config)
        self.real_hidden_size = config['real_hidden_size']
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(config['real_hidden_size'], config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.attention_hops = config['attention-hops']
        self.attention_pen = config['penalization_coeff']

        # Classifier
        self.fc_clf = nn.Linear(config['real_hidden_size'] * config['attention-hops'], config['nfc'])
        self.pred_clf = nn.Linear(config['nfc'], config['class-number'])
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def __str__(self):
        s = ''
        s += str(self.convlstm)
        return s

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(path)

    def forward(self, x: torch.tensor):
        """
        x,                 features = [B, Strands, Chromosomes, Sequence Length]
        """

        _, meta_results = self.convlstm(x)
        hidden = torch.stack(meta_results['hidden'], dim=1)
        size = hidden.size()                                            # [batch_size, num_multiscales, n_hidden]

        compressed_embeddings = hidden.view(-1, size[2])

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))   # [batch_size, num_multiscales, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)             # [batch_size, num_multiscales, attention-hops]
        alphas = torch.transpose(alphas, 1, 2).contiguous()            # [batch_size, attention-hops, num_multiscales]

        alphas = self.softmax(alphas.view(-1, size[1]))                # [batch_size * attention-hops, num_multiscales]
        alphas = alphas.view(size[0], self.attention_hops, size[1])    # [batch_size, attention-hops, num_multiscales]

        output = torch.bmm(alphas, hidden)                             # [batch_size, attention-hops, n_hidden]
        meta_results["attention"] = alphas                             # A in Bengio paper
        meta_results["M"] = output

        output = output.view(output.size(0), -1)                       # [batch_size, attention-hops * n_hidden]
        fc = self.tanh(self.fc_clf(self.drop(output)))
        pred = self.pred_clf(self.drop(fc))

        return pred, meta_results

    def loss_function(self, targets, pred_t, meta) -> dict:

        bsz = meta["attention"].size(0)          # batch size

        # Classifier loss
        clf_loss = self.criterion(pred_t, targets)

        # Calculate penalization to ensure diversity in hops
        atn_t = torch.transpose(meta["attention"], 1, 2).contiguous()
        eye = torch.stack([torch.eye(self.attention_hops) for _ in range(bsz)], dim=0)
        p = torch.norm(torch.bmm(meta["attention"], atn_t) - eye.to(device))
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
        self.log("test_acc", test_loss_dict['acc'], prog_bar=True, logger=True)
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


def create_classifier(classes, seq_length, strands, chromosomes,
                      hidden_size=256, layers=1, proj_size=0,
                      wavelet='haar',
                      coarse_skip=0,
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
                      "coarse_skip": coarse_skip,
                      "recursion_limit": recursion_limit
                      }
    config = {"dropout": 0.5,
              "attention-unit": 350,
              "attention-hops": 10 if recursion_limit is None else recursion_limit,
              "penalization_coeff": 0.,
              "nfc": 128,                         # hidden layer size for MLP in the classifier
              "class-number": len(classes),       # number of class for the last step of classification
              "real_hidden_size": proj_size if proj_size > 0 else hidden_size,       # Encoder's hidden size
              }

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
        monitor="val_clf_loss", mode="min",
        min_delta=0,
        patience=5,
        verbose=verbose
    )

    viz_embedding_callback = ViewEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    viz_rnn_callback = ViewRecurrentSignal(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    viz_attention_1 = MultiScaleEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch,
        class_labels=classes
    )

    callbacks = [checkpoint_callback,
                 early_stop_callback,
                 viz_rnn_callback,
                 viz_embedding_callback,
                 viz_attention_1
                 ]

    _trainer = pl.Trainer(
        default_root_dir=dir_path,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        max_epochs=num_epochs,
        check_val_every_n_epoch=2,
        gpus=gpus,
    )

    return _model, _trainer
