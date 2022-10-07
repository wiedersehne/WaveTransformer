import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceEncoder(nn.Module):

    def __init__(self,
                 out_dim: int,
                 n_hidden: int = 256,
                 n_layers: int = 3,
                 dropout: float = 0.6,
                 bidirectional: bool = False,
                 stack: int = 1,
                 in_channels: int = 23,
                 out_channels: int = 23,
                 kernel_size: int = 3,
                 stride: int = 5,
                 padding: int = 1):
        """

        :param out_dim:
        :param n_hidden:
        :param n_layers:
        :param dropout:
        :param bidirectional:
        :param stack:                           Dimension to stack chromosomes on
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        """
        super().__init__()
        self.hidden_size = n_hidden
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.stack_dim = stack                          # 1 => Major/min separated by feature. 2=> sequences appended

        self.conv1d_major = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.conv1d_minor = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.lstm = nn.LSTM(
            input_size=out_channels*2,  # int((seq_length/stride)*self.stack_dim),
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.activation = torch.nn.ReLU()
        if bidirectional is False:
            self.fc = nn.Linear(n_hidden, out_dim)
        else:
            self.fc = nn.Linear(2 * n_hidden, out_dim)

    def forward(self, x):
        # Take in batch_size x num_strands x num_chromosomes x sequence_length

        # Split sequences
        # Independent CNN networks for each sequence,
        x_strand1 = self.conv1d_major(x[:, 0, :, :])                   # chromosomes are channels of CNN,
        x_strand2 = self.conv1d_minor(x[:, 1, :, :])                   # and length gets down-sampled from striding

        # Concatenate either along sequence (dim=2) or feature dimension (dim=1)
        x = torch.cat([x_strand1, x_strand2], dim=self.stack_dim)  # Stack output channels of CNN as features to LSTM
        x = torch.transpose(x, 1, 2)

        # LSTM layers
        self.lstm.flatten_parameters()                         # For multiple GPU cases

        d = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.n_layers * d, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers * d, x.size(0), self.hidden_size).to(device)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        if self.bidirectional:
            out = out[:, -1, :]
        else:
            out = hn[-1]

        out = self.fc(out)

        return out
