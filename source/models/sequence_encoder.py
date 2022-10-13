import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceEncoder(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_hidden: int = 256,
                 n_layers: int = 3,
                 dropout: float = 0.6,
                 bidirectional: bool = False,
                 in_channels: int = 2,
                 out_channels: int = 2,
                 kernel_size: int = 3,
                 stride: int = 5,
                 padding: int = 1):
        """

        :param out_dim:
        :param n_hidden:
        :param n_layers:
        :param dropout:
        :param bidirectional:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        """
        super().__init__()
        self.hidden_size = n_hidden
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.convolution_strands = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.lstm = nn.LSTM(
            input_size=int(np.ceil(in_features/stride)),
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.activation = torch.nn.ReLU()
        if bidirectional is False:
            self.fc = nn.Linear(n_hidden, out_features)
        else:
            self.fc = nn.Linear(2 * n_hidden, out_features)

    def forward(self, x):
        # Take in batch_size x num_strands x num_chromosomes x sequence_length

        # CNN network, with each sequence as a channel
        c_out = self.convolution_strands(x)

        # LSTM layers
        self.lstm.flatten_parameters()                         # For multiple GPU cases
        d = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.n_layers * d, c_out.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers * d, c_out.size(0), self.hidden_size).to(device)
        out, (hn, cn) = self.lstm(c_out, (h0, c0))

        if self.bidirectional:
            rnn_out = out[:, -1, :]
        else:
            rnn_out = hn[-1]

        out = self.fc(rnn_out)
        # print(f"in: {x.shape}, conv_out: {c_out.shape}, and LSTM out: {rnn_out.shape}, fc out {out.shape}")

        return out