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

from WaveLSTM.custom_callbacks import clf_callbacks
from WaveLSTM.custom_callbacks import waveLSTM_callbacks
from WaveLSTM.modules.encoder import Encoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelfAttentiveEncoder(pl.LightningModule, ABC):

    def __init__(self,
                 encoder_config,
                 config,
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(**encoder_config)
        self.real_hidden_size = config['real_hidden_size']
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.LazyLinear(config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.init_weights(init_range=0.1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.attention_hops = config['attention-hops']
        self.attention_pen = config['penalization_coeff']

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def __str__(self):
        s = ''
        s += str(self.encoder)
        return s

    def forward(self, x: torch.tensor):
        """
        x,                 features = [B, Strands, Chromosomes, Sequence Length]
        """
        assert x.dim() == 4

        hidden, meta_results = self.encoder(x)
        meta_results.update({'hidden': hidden})

        hidden = torch.stack(hidden, dim=1)
        size = hidden.size()                                            # [batch_size, num_multiscales, n_hidden]
        hops = self.attention_hops

        compressed_embeddings = hidden.view(-1, size[2])

        hbar = self.tanh(self.ws1(compressed_embeddings))              # [batch_size, num_multiscales, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)             # [batch_size, num_multiscales, attention-hops]
        alphas = torch.transpose(alphas, 1, 2).contiguous()            # [batch_size, attention-hops, num_multiscales]

        alphas = self.softmax(alphas.view(-1, size[1]))                # [batch_size * attention-hops, num_multiscales]
        alphas = alphas.view(size[0], self.attention_hops, size[1])    # [batch_size, attention-hops, num_multiscales]

        # M in Bengio's paper
        output = torch.bmm(alphas, hidden)                             # [batch_size, attention-hops, n_hidden]

        meta_results.update({"attention": alphas,                      # A in Bengio's self-attention paper
                             })

        return output, meta_results
