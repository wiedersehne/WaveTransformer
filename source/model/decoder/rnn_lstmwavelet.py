import torch
import torch.nn as nn
from source.model.encoder.sequence_encoder import SequenceEncoder
from source.model.decoder.Conv1dLSTM import Conv1dLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WaveletConv1dLSTM(nn.Module):

    def __str__(self):
        s = '\nWaveletLSTM'
        s += f'\n\t Number of directions in LSTM cells {self.directions}'
        s += f'\n\t Number of layers {self.lstm_layers}'
        s += f'\n\t Hidden/cell sizes {self.hidden_size}'
        s += f'\n\t {self.channels} channels, of length {self.width}'
        return s

    def __init__(self, out_features: int, strands: int, chromosomes: int,
                 hidden_size,
                 layers: int = 1,
                 init="learn",
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
        self.bidirectional, self.directions = False, 1
        self.C, self.H, self.W = strands, chromosomes, out_features     # NCHW format
        self.convolutional = True

        self.width = self.W
        self.channels = self.C * self.H
        self.pool = torch.nn.MaxPool1d(3, stride=2)

        def pool_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
            return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        self.pooled_width = pool_output_length(self.width, 3, stride=2)

        # LSTM
        if init == "learn":
            self.h0_network = SequenceEncoder(in_features=self.H * self.W, out_features=5,
                                              in_channels=2, out_channels=2)
            self.h0_hidden = nn.Linear(5, self.lstm_layers * self.directions * self.real_hidden_size * self.W)
            self.h0_cell = nn.Linear(5, self.lstm_layers * self.directions * self.hidden_size * self.W)

        self.rnn = Conv1dLSTM(input_size=self.channels,
                              hidden_size=self.hidden_size,
                              kernel_size=3,
                              num_layers=self.lstm_layers,
                              proj_size=self.proj_size
                              )
        # If we use the ConvLSTM, we need to reduce to 1 channel before putting through coeff net.
        conv_list = [nn.Conv1d(in_channels=self.W,
                               out_channels=1,
                               kernel_size=3,
                               ) for _ in range(20)]     # todo: create correct number of nets
        self.conv_list = nn.ModuleList(conv_list)

    def forward(self, x, hidden_state, t):
        """
        x,             [batch size, strands, chromosomes, seq_length]
        hidden,        [n layers * n directions, batch size, hid dim]
        cell,          [n layers * n directions, batch size, hid dim]
        """
        # Determine which dimensions are used as input channels
        x = x.view((x.size(0), self.channels, -1))
        # x = self.pool(x)

        # print(f"len{len(hidden_state)}, type{len(hidden_state[0])}, type({hidden_state[0][0].shape}")
        output, (h_next, c_next) = self.rnn(x, hidden_state)
        # output: (N, L, H_out, Signal length)

        if self.bidirectional:
            raise NotImplementedError
        else:
            output = output[:, -1, :, :].permute(0, 2, 1)  # Take the last of `temporal` seq
            latent = self.conv_list[t](output).squeeze(1)
            coeff = self.coeff_nets[t](latent)

        return coeff, (h_next, c_next), latent

    def init_states(self, batch, method='zeros'):
        batch_size = batch.size(0)
        if method == 'zeros':
            hidden = torch.zeros((self.lstm_layers * self.directions,
                                  batch_size,
                                  self.real_hidden_size,
                                  self.W), device=device)
            cell = torch.zeros((self.lstm_layers * self.directions,
                                batch_size,
                                self.hidden_size,
                                self.W), device=device)
        elif method == 'learn':
            # We can also learn the initial states, which often improves training speed
            z = self.h0_network(batch)
            hidden = torch.reshape(self.h0_hidden(z), (batch_size,
                                                       self.lstm_layers * self.directions,
                                                       self.real_hidden_size,
                                                       -1)).permute((1, 0, 2, 3)).contiguous()
            cell = torch.reshape(self.h0_cell(z), (batch_size,
                                                   self.lstm_layers * self.directions,
                                                   self.hidden_size,
                                                   -1)).permute((1, 0, 2, 3)).contiguous()
        else:
            raise NotImplementedError

        return hidden, cell
