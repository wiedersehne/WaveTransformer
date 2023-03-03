import torch
import torch.nn as nn
from source.model.encoder.Conv1dLSTM import Conv1dLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WaveletConv1dLSTM(nn.Module):

    @staticmethod
    def pool_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
        return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def __str__(self):
        s = '\nWaveletLSTM'
        s += f'\n\t Number of layers {self.lstm_layers}'
        s += f'\n\t Cell size {self.hidden_size}, hidden size {self.real_hidden_size}'
        s += f'\n\t {self.channels} channels, of length {self.W}'
        return s

    def __init__(self, out_features: int, strands: int, chromosomes: int,
                 bank_lengths,
                 hidden_size,
                 layers: int = 1,
                 proj_size: int = 0,
                 ):
        """
        out_features,  number of loci
        strands,       number of strands (=2 in example cases)
        chromosomes,   number of chromosomes (=23 in example cases)
        hidden_size,       dimension of the hidden and cell states of the LSTM
        n_layers,      the number of LSTM layers
        """
        super().__init__()
        self.proj_size = proj_size
        self.hidden_size = hidden_size
        self.real_hidden_size = proj_size if proj_size > 0 else hidden_size
        self.lstm_layers = layers
        self.C, self.H, self.W = strands, chromosomes, out_features     # NCHW format

        self.channels = self.C * self.H

        # Pooling input layer
        self.pool = torch.nn.MaxPool1d(3, stride=2)
        self.pooled_width = self.pool_output_length(self.W, 3, stride=2)

        # LSTM
        self.rnn = Conv1dLSTM(input_size=self.channels,
                              hidden_size=self.hidden_size,
                              kernel_size=3,
                              num_layers=self.lstm_layers,
                              proj_size=self.proj_size
                              )

        # Output layers
        # CNN to reduce to 1 channel before putting through network to predict coefficients.
        conv_list = [nn.Conv1d(in_channels=self.pooled_width,
                               out_channels=1,
                               kernel_size=3,
                               padding=1,
                               ) for _ in range(len(bank_lengths))]
        self.conv_list = nn.ModuleList(conv_list)
        # Connect reduced channel to wavelet coefficient
        self.coeff_nets = [nn.Linear(in_features=self.real_hidden_size, out_features=length, device=device)
                           for length in bank_lengths]
        self.coeff_nets = nn.ModuleList(conv_list)

    def forward(self, x, hidden_state, t):
        """
        x,             [batch size, strands, chromosomes, seq_length]
        hidden,        [n layers, batch size, hid dim]
        cell,          [n layers, batch size, hid dim]
        """
        # Use both strands and chromosomes as channels
        x = x.view((x.size(0), self.channels, -1))

        # Pooling input layer, reducing the width W of the signals (similar function to a word embedding)
        x = self.pool(x)

        # Recurrent
        output, (h_next, c_next) = self.rnn(x, hidden_state)        # output: (N, L, H_out, Signal length)

        # Output layer
        output = output[:, -1, :, :].permute(0, 2, 1)               # Take the last of `temporal` seq
        latent = self.conv_list[t](output).squeeze(1)
        coeff = self.coeff_nets[t](latent)

        return coeff, (h_next, c_next), latent

    def init_states(self, batch):
        return (torch.zeros((self.lstm_layers, batch.size(0), self.real_hidden_size, self.W), device=device),
                torch.zeros((self.lstm_layers, batch.size(0), self.hidden_size, self.W), device=device))
