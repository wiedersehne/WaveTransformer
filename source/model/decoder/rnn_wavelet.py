import torch
import torch.nn as nn
from source.model.encoder.sequence_encoder import SequenceEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WaveletLSTM(nn.Module):

    @property
    def real_hidden_size(self):
        return self.proj_size if self.proj_size > 0 else self.hidden_size

    def __str__(self):
        s = '\nWaveletLSTM'
        s += f'\n\t Number of directions in LSTM cells {self.directions}'
        s += f'\n\t Number of layers {self.lstm_layers}'
        s += f'\n\t Hidden/cell sizes {self.hidden_size}, with hidden projection to {self.proj_size} dimensions'
        s += f'\n\t {self.H} chromosomes, of length {self.W} and {self.C} channels'
        return s

    def __init__(self, out_features: int, strands: int, chromosomes: int,
                 hidden_size: int,
                 layers: int,
                 bidirectional: bool,
                 proj_size=0,
                 dropout=0.6,
                 init="learn",
                 ):
        """
        out_features,  number of loci
        strands,       number of strands (=2 in example cases)
        chromosomes,   number of chromosomes (=23 in example cases)
        hidden_size,       dimension of the hidden and cell states of the LSTM
        n_layers,      the number of LSTM layers
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.lstm_layers = layers
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        self.C, self.H, self.W = strands, chromosomes, out_features     # NCHW format
        self.L = self.C * self.H * self.W
        self.proj_size = proj_size
        self.convolutional = False

        self.test_stack = None
        if self.test_stack is None:
            # no sequence inside rec cell, everything is concatenated
            h_in = self.C * self.H * self.W
        elif self.test_stack == "strands":
            # just strands splitting into LSTM sequence
            h_in = self.H * self.W
        elif self.test_stack == "chr":
            # just chromosomes splitting into LSTM sequence
            h_in = self.C * self.W
        elif self.test_stack == "features":
            # sequence is along out_features
            h_in = self.C * self.H
        elif self.test_stack == "both":
            # chromosomes and strands splitting into LSTM sequence
            h_in = self.W
        else:
            raise NotImplementedError

        # LSTM
        if init == "learn":
            self.h0_network = SequenceEncoder(in_features=self.H * self.W, out_features=5, in_channels=2,  out_channels=2)
            self.h0_hidden = nn.Linear(5, self.lstm_layers * self.directions * self.real_hidden_size)
            self.h0_cell = nn.Linear(5, self.lstm_layers * self.directions * self.hidden_size)

        self.rnn = nn.LSTM(input_size=h_in,
                           hidden_size=self.hidden_size,
                           num_layers=self.lstm_layers,
                           proj_size=self.proj_size,
                           batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           )

    def forward(self, x, hidden_state, t):
        """
        x,             [batch size, strands, chromosomes, seq_length]
        hidden,        [n layers * n directions, batch size, hid dim or proj_size]
        cell,          [n layers * n directions, batch size, hid dim or proj_size]
        """
        if self.test_stack is None:
            x = x.view((x.size(0), 1, -1))                     # Stack all CHW dimensions
        elif self.test_stack == "strands":
            pass
        elif self.test_stack == "chr":
            x = x.permute((0, 2, 1, 3)).contiguous()
        elif self.test_stack == "features":
            x = x.permute((0, 3, 1, 2)).contiguous()
        elif self.test_stack == "both":
            x = x.view((x.size(0), x.size(1) * x.size(2), -1))
        x = x.view((x.size(0), x.size(1), -1))

        output, (hidden, cell) = self.rnn(x, hidden_state)
        # output = [B, seq_len=1, H_out * num_directions]
        # hidden = [layers * num_directions, B, H_out]
        # cell   = [layers * num_directions, B, H_out]

        if self.bidirectional:
            # output = output.reshape((output.shape[0], -1))   # Stack final forward and initial reverse for both strands
            latent = output[:, -1, :]  # Take the the last seq (so last strand - or (TODO: chromosome)), and map to the output of all
            coeff = self.coeff_nets[t](latent)

        else:
            raise NotImplementedError
            output = output.squeeze(1)

        return coeff, (hidden, cell), latent

    def init_states(self, batch, method='learn'):
        batch_size = batch.size(0)
        if method == 'zeros':
            hidden = torch.zeros((self.lstm_layers * self.directions,
                                  batch_size,
                                  self.real_hidden_size), device=device)
            cell = torch.zeros((self.lstm_layers * self.directions,
                                batch_size,
                                self.hidden_size), device=device)
        elif method == 'learn':
            # We can also learn the initial states, which often improves training speed
            z = self.h0_network(batch)
            hidden = torch.reshape(self.h0_hidden(z), (batch_size,
                                                       self.lstm_layers * self.directions,
                                                       self.real_hidden_size)).permute((1, 0, 2)).contiguous()
            cell = torch.reshape(self.h0_cell(z), (batch_size,
                                                   self.lstm_layers * self.directions,
                                                   self.hidden_size)).permute((1, 0, 2)).contiguous()
        else:
            raise NotImplementedError

        return hidden, cell
