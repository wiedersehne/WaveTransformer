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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentiveClassifier(pl.LightningModule, ABC, WaveletBase):

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

        # Encoder
        encoder_config["J"] = self.J
        encoder_config["pooled_width"] = self.masked_width
        self.a_encoder = SelfAttentiveEncoder(encoder_config=encoder_config,
                                              config=attention_config)

        # Classifier network
        self.clf_net = nn.Sequential(nn.LazyLinear(config['nfc']),
                                     nn.Tanh(),
                                     nn.Dropout(config['dropout']),
                                     nn.LazyLinear(config['nfc']),
                                     nn.Tanh(),
                                     nn.Dropout(config['dropout']),
                                     nn.LazyLinear(len(config['classes'])))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.tensor):

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
        output = output.view(output.size(0), -1)                        # [batch_size, attention-hops * scale_embed_dim]

        # Decode
        pred = self.clf_net(output)

        return pred, meta_data

    def loss_function(self, targets, pred_t, meta) -> dict:

        bsz = targets.shape[0]                   # batch size
        atn = meta["attention"]

        # Classifier loss
        clf_loss = self.criterion(pred_t, targets)

        # Calculate penalization (promotes diversity in hops)
        # atn_t = torch.transpose(atn, 1, 2).contiguous()
        # eye = torch.stack([torch.eye(self.a_encoder.attention_hops) for _ in range(bsz)], dim=0)
        # p = torch.norm(torch.bmm(atn, atn_t) - eye.to(device))
        # penalty = self.a_encoder.attention_pen * p

        # Record classification accuracy
        _, pred_label = torch.max(pred_t, dim=1)
        acc = torch.sum(pred_label == targets).item() / bsz

        return {'loss': clf_loss, #+ penalty,              # Penalized loss
                'acc': acc}                              # Accuracy

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
            "frequency": 2,                     # How many epochs/steps should pass between calls to `scheduler.step()`
            "monitor": "val_loss",              # Metric to monitor for scheduler
            "strict": True,                     # Enforce that "val_loss" is available when the scheduler is updated
            "name": 'LearningRateMonitor',      # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }



def create_classifier(classes, seq_length, channels,
                      hidden_size=256, layers=1, proj_size=0, scale_embed_dim=128,
                      r_hops=1, clf_nfc=128,
                      wavelet='haar',
                      recursion_limit=None,
                      dir_path="logs", verbose=False, monitor="val_loss",
                      num_epochs=100, gpus=1,
                      validation_hook_batch=None, test_hook_batch=None,
                      project='WaveLSTM-clf', run_id="null",
                      outfile="logs/clf-output.pkl"
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
    clf_decoder_config = {"dropout": 0.0,
                          "nfc": clf_nfc,                            # hidden layer size for MLP in the classifier
                          "classes": classes,                   # number of class for the last step of classification
                          }

    _model = AttentiveClassifier(encoder_config=encoder_config,
                                 attention_config=attention_config,
                                 config=clf_decoder_config
                                 )
    print(_model)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project=project,
                               name=run_id,
                               job_type='train',
                               save_dir=dir_path
                               )

    # Make all callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_path + "/checkpoints",
        filename=run_id,
        verbose=verbose,
        monitor=monitor,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", mode="min",
        min_delta=0,
        patience=10,
        verbose=verbose
    )

    label_dictionary = {key: val for key, val in zip([i for i in range(len(classes))], classes)}

    viz_embedding_callback = waveLSTM.ResolutionEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch,
        label_dictionary=label_dictionary
    )

    viz_multi_res_embed = attention.MultiResolutionEmbedding(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch,
        label_dictionary=label_dictionary
    )

    viz_attention = attention.Attention(
        val_samples=validation_hook_batch,
        test_samples=test_hook_batch
    )

    save_output = waveLSTM.SaveOutput(
        test_samples=test_hook_batch,
        file_path=outfile
    )

    callbacks = [checkpoint_callback,
                 early_stop_callback,
                 viz_embedding_callback,
                 viz_multi_res_embed,
                 viz_attention,
                 save_output
                 ]

    _trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=num_epochs,
        check_val_every_n_epoch=2,
        gpus=gpus,
        log_every_n_steps=5
    )

    return _model, _trainer
