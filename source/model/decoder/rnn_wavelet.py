import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WaveletLSTM(nn.Module):

    @property
    def real_hidden_size(self):
        return self.proj_size if self.proj_size > 0 else self.hidden_size

    @staticmethod
    def calculate_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
        return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def __str__(self):
        s = '\nWaveletLSTM'
        s += f'\n\t Number of directions in LSTM cells {2 if self.bidirectional else 1}'
        s += f'\n\t Number of layers {self.lstm_layers}'
        s += f'\n\t Hidden/cell sizes {self.hidden_size}, with hidden projection to {self.proj_size} dimensions'
        s += f'\n\t {self.H} chromosomes, of length {self.W} and {self.C} channels'
        if self.conv is not None:
            s += f'\n\tConvolving the {self.C} channels before LSTM cells'
        return s

    def __init__(self, out_features: int, strands: int, chromosomes: int,
                 hidden_size: int,
                 layers: int,
                 bidirectional: bool,
                 proj_size=0,
                 dropout=0,
                 conv_layer=False,
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
        self.C, self.H, self.W = strands, chromosomes, out_features     # NCHW format
        self.L = strands * chromosomes * out_features   # flat sequence length
        self.proj_size = proj_size

        if conv_layer:
            kernel_size = 1
            stride = 1
            self.conv = nn.Conv1d(in_channels=strands,  # * chromosomes,
                                  out_channels=1,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  )
            h_in = self.calculate_output_length(self.W * self.H, kernel_size=kernel_size, stride=stride)
        else:
            self.conv = None
            h_in = self.L

        self.rnn = nn.LSTM(input_size=h_in,
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
        # print(x.shape)
        if self.conv is not None:
            # x = x.reshape((x.size(0), self.C * self.H, self.W))    # stack strands along chromosome index
            x = x.reshape((x.size(0), self.C, self.H * self.W))      # stack strands, concat chromosome along signal
            x_in = self.conv(x)
        else:
            x_in = x.reshape((x.size(0), -1)).unsqueeze(1)     # Stack all CHW dimensions

        output, (hidden, cell) = self.rnn(x_in, (hidden, cell))
        # print(f"{self.rnn}:"
        #       f" \n\t-> in: x: {x.shape}  x_in {x_in.shape}"
        #       f" \n\t-> out: {output.shape}, {hidden.shape}, {cell.shape}")

        # TODO: For bidirectional LSTMs, h_n is not equivalent to the last element of output; the former contains the final forward and reverse hidden states, while the latter contains the final forward hidden state and the initial reverse hidden state.
        # output = [B, seq_len=1, H_out * num_directions]
        # hidden = [layers * num_directions, B, H_out]
        # cell   = [layers * num_directions, B, H_out]
        if self.bidirectional:
            return output.squeeze(1), (hidden, cell)
        else:
            return output.squeeze(1), (hidden, cell)

