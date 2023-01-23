import torch
import torch.nn as nn
import numpy as np
from source.model.encoder.sequence_encoder import SequenceEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WaveletLSTM(nn.Module):

    def pooling_layer(self, x):
        if self.embed == 'SequenceEncoder':
            x_emb = self.embed_layer(x)
            return x_emb.reshape((x_emb.size(0), -1)).unsqueeze(1)
        else:
            x_emb = self.embed_layer(x.reshape(x.shape[0], -1))
            return x_emb.reshape((x_emb.size(0), -1)).unsqueeze(1)

    @property
    def real_hidden_size(self):
        return self.emb_dim if self.embed == "proj" else self.h_dim

    def __init__(self, out_features: int, strands: int, chromosomes: int,
                 hid_dim: int,
                 layers: int,
                 bidirectional: bool,
                 embedding='AvgPool1d',
                 ):
        """
        out_features,  number of loci
        strands,       number of strands (=2 in our genomics cases)
        chromosomes,   number of chromosomes (=23)
        emb_dim,       latent embedding dimension dim(z)
        hid_dim,       dimension of the hidden and cell states of the LSTM
        n_layers,      the number of LSTM layers
        dropout,       the amount of dropout to regularize our LSTM
        """
        super().__init__()

        self.h_dim = hid_dim
        self.lstm_layers = layers
        self.bidirectional = bidirectional
        self.L = strands * chromosomes * out_features   # flat sequence length
        self.embed = embedding

        # Pooling
        if self.embed == 'SequenceEncoder':
            self.emb_dim = 10
            self.embed_layer = SequenceEncoder(in_features=out_features, out_features=self.emb_dim,
                                               n_hidden=64, n_layers=1,
                                               dropout=0., bidirectional=True, in_channels=2, out_channels=2,
                                               kernel_size=3, stride=5, padding=1)
        elif self.embed == 'AvgPool1d':
            kernel_size, stride, padding = int(np.floor(self.L / 5)), int(np.floor(self.L / 5)), 0
            self.embed_layer = torch.nn.AvgPool1d(kernel_size, stride=stride)
            self.emb_dim = int(np.floor(1 + (self.L - kernel_size + (2*padding)) / stride))
        elif self.embed == 'Id':
            self.embed_layer = nn.Identity()
            self.emb_dim = self.L
        elif self.embed == 'W':
            self.embed_layer = nn.Linear(self.L, 10)
            self.emb_dim = 10
        # elif self.embed == 'proj':
        #     self.embed_layer = nn.Identity()
        #     self.emb_dim = 10
        # else:
        #     raise NotImplementedError

        self.rnn = nn.LSTM(input_size=self.L if self.embed == "proj" else self.emb_dim,
                           hidden_size=self.h_dim,
                           num_layers=self.lstm_layers,
                           proj_size=self.emb_dim if self.embed == 'proj' else 0,
                           bias=True,
                           batch_first=True,
                           bidirectional=bidirectional,
                           dropout=0.2
                           )

    def forward(self, x, hidden, cell):
        """
        x,             [batch size, strands, chromosomes, seq_length, H_in=1]
        hidden,        [n layers * n directions, batch size, hid dim or proj_size]
        cell,          [n layers * n directions, batch size, hid dim or proj_size]
        """

        output, (hidden, cell) = self.rnn(self.pooling_layer(x), (hidden, cell))
        # print(f"{self.rnn}: in: {x.shape} -> out: {output.shape}, {hidden.shape}, {cell.shape}")

        if self.bidirectional:
            # output = [B, seq_len=1, H_out * 2]
            # hidden = [layers * 2, B, H_out]
            # cell   = [layers * 2, B, H_out]
            return output.squeeze(1), (hidden, cell)
        else:
            # output = [B, seq_len=1, H_out * 1]
            # hidden = [layers * 1, B, H_out]
            # cell   = [layers * 1, B, H_out]
            return output.squeeze(1), (hidden, cell)
