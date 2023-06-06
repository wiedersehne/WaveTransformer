import torch
import torch.nn as nn
from WaveLSTM.modules.Conv1dLSTM import Conv1dLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WaveletConv1dLSTM(nn.Module):

    def __str__(self):
        s = '\nWaveletLSTM'
        s += f'\n\t Number of layers {self.lstm_layers}'
        s += f'\n\t Cell state size {self.hidden_size}, hidden state size {self.real_hidden_size}'
        s += f'\n\t {self.channels} channels'
        s += f"\n\t Masked input width {self.masked_input_width}"
        return s

    def __init__(self, out_features: int, channels: int,
                 masked_input_width: int,
                 J: int,
                 hidden_size,
                 layers: int = 1,
                 proj_size: int = 0,
                 scale_embed_dim: int = 128,
                 dropout=0.5
                 ):
        """
        out_features,  number of loci     # TODO: still needed?
        strands,       number of strands (=2 in example cases)
        chromosomes,   number of chromosomes (=23 in example cases)
        hidden_size,       dimension of the hidden and cell states of the LSTM
        n_layers,      the number of LSTM layers
        """
        super().__init__()
        # self.permute = False
        self.proj_size = proj_size
        self.hidden_size = hidden_size
        self.real_hidden_size = proj_size if proj_size > 0 else hidden_size
        self.lstm_layers = layers
        self.embed_dim = scale_embed_dim

        self.channels = channels
        self.masked_input_width = masked_input_width

        # LSTM
        self.rnn = Conv1dLSTM(input_channels=self.channels,
                              hidden_channels=self.hidden_size,
                              kernel_size=3,
                              num_layers=self.lstm_layers,
                              proj_size=self.proj_size,
                              dropout=dropout
                              )

        # Output layers to map to the embedded space
        conv_list = [nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=self.embed_dim),
        )
            for _ in range(J)]
        self.conv_list = nn.ModuleList(conv_list)

    def forward(self, x, hidden_state, t):
        """
        x,             [batch size, strands, chromosomes, seq_length]
        hidden,        [n layers, batch size, hid dim]
        cell,          [n layers, batch size, hid dim]
        """

        x = x.view((x.size(0), self.channels, -1))

        # Recurrent
        output, (h_next, c_next) = self.rnn(x, hidden_state)        # output: (N, L, H_out, Signal length)

        # Output layer
        output = output[:, -1, :, :]                                # Take the last of `temporal` seq
        scale_embedding = self.conv_list[t](output)           #.squeeze(1)

        return scale_embedding, (h_next, c_next)

    def init_states(self, batch_size):

        return (torch.zeros((self.lstm_layers,
                             batch_size,
                             self.real_hidden_size,
                             self.masked_input_width), device=device),
                torch.zeros((self.lstm_layers,
                             batch_size,
                             self.hidden_size,
                             self.masked_input_width), device=device))
