import torch
import torch.nn as nn
import numpy as np
from source.model.encoder.sequence_encoder import SequenceEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WaveletLSTM(nn.Module):

    @property
    def real_hidden_size(self):
        return self.proj_size if self.proj_size > 0 else self.hidden_size

    def __init__(self, out_features: int, strands: int, chromosomes: int,
                 hidden_size: int,
                 layers: int,
                 bidirectional: bool,
                 proj_size=0,
                 dropout=0,
                 ):
        """
        out_features,  number of loci
        strands,       number of strands (=2 in our genomics cases)
        chromosomes,   number of chromosomes (=23)
        hidden_size,       dimension of the hidden and cell states of the LSTM
        n_layers,      the number of LSTM layers
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.lstm_layers = layers
        self.bidirectional = bidirectional
        self.L = strands * chromosomes * out_features   # flat sequence length
        self.proj_size = proj_size

        self.rnn = nn.LSTM(input_size=self.L,
                           hidden_size=self.hidden_size,
                           num_layers=self.lstm_layers,
                           proj_size=self.proj_size,
                           batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           )

    def forward(self, x, hidden, cell):
        """
        x,             [batch size, strands, chromosomes, seq_length, H_in=1]
        hidden,        [n layers * n directions, batch size, hid dim or proj_size]
        cell,          [n layers * n directions, batch size, hid dim or proj_size]
        """

        x = x.reshape((x.size(0), -1)).unsqueeze(1)
        output, (hidden, cell) = self.rnn(x, (hidden, cell))
        # print(f"{self.rnn}: in: {x.shape} -> out: {output.shape}, {hidden.shape}, {cell.shape}")

        # TODO: For bidirectional LSTMs, h_n is not equivalent to the last element of output; the former contains the final forward and reverse hidden states, while the latter contains the final forward hidden state and the initial reverse hidden state.
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
