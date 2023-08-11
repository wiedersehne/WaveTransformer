import torch
import torch.nn as nn
from WaveLSTM.modules.Conv1dLSTM import Conv1dLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WaveletConv1dLSTM(nn.Module):
    r"""A wrapper around the Conv1dLSTM module.

    This class performs the non-recurrent projections from the reccurrent hidden state tensors, from the last layer of
    the Conv1dLSTM network.

    For each element from the sequence of hidden state tensors produced by the Conv1dLSTM network,
    :math:`\{\H_j^{\left(L\right)}\}_{j=1}^J`, this module computes the following linear, non-recurrent projection:

    .. math::
        \begin{array}{ll} \\
            h_j &= W_{j}F(H_j^{\left(L\right)})
        \end{array}

    where :math:`H_j^{\left(L\right)}` is the hidden state tensor at resolution `j`, belonging to the last layer of the
    LSTM network, :math:`F` denotes a flattening layer, and :math:`W_{j}` is the linear projection matrix.


    Args:
        input_channels (int):
            The number of channels of the input data.
        hidden_size (int):
            The number of channels of the cell state. Default ``128``.
        kernel_size (int):
            Size of the convolutional kernel. Default ``3``.
        bias (bool):
            Whether to add the bias to convolutions. Default ``True``.
        proj_size (int):
            If ``>0``, will use ConvLSTM with hidden state projections with corresponding number of channels. Default ``0``.
        drropout (float):
            If non-zero, introduces a Dropout layer on the outputs of the Conv1dLSTM cell except the last layer,
             with dropout probability equal to dropout. Default: 0


    Inputs: input_tensor, (h_0, c_0)
        * **input_tensor**: tensor of shape :math:`(N, J, H_{in}, W)`
        * **h_0**: tensor of shape :math:`(\text{num\_layers}, N, H_{proj}, W)`
        * **c_0**: tensor of shape :math:`(\text{num\_layers}, N, H_{cell}, W)`

    Outputs: output, (h_n, c_n)
        * **output**: tensor of shape :math:`(N, J, H_{proj}, W)` containing the hidden state tensor features from the
         last layer of the LSTM, for each resolution
        * **h_n**: List of L tensors of shape :math:`(N, H_{proj}, W)` containing the final (of J) hidden state
         tensor for each element in the sequence
        * **c_n**: List of L tensor of shape :math:`(N, H_{cell}, W)` containing the final (of J) cell state tensor
         for each element in the sequence

    """
    def __str__(self):
        s = '\nWaveletLSTM'
        s += str(self.rnn)
        s += '\nNon-recurrent projection'
        s += f'\n\t {self.channels} channels'
        s += f"\n\t Masked input width {self.masked_input_width}"
        return s

    def __init__(self,
                 out_features: int,
                 channels: int,
                 masked_input_width: int,
                 J: int,
                 hidden_size,
                 layers: int = 1,
                 proj_size: int = 0,
                 scale_embed_dim: int = 128,
                 kernel_size: int = 7,
                 dropout_input=0, # TODO: re
                 dropout_hidden=0,   # TODO: remove
                 dropout_proj=0      # TODO: remove
                 ):
        """
        out_features,  number of loci     # TODO: still needed?
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
        self.embed_dim = scale_embed_dim
        self.kernel_size=kernel_size

        self.channels = channels
        self.masked_input_width = masked_input_width

        # LSTM
        self.rnn = Conv1dLSTM(input_channels=self.channels,
                              hidden_channels=self.hidden_size,
                              kernel_size=self.kernel_size,
                              num_layers=self.lstm_layers,
                              proj_size=self.proj_size,
                              dropout=dropout_input,
                              )

        # Output layers to map to the embedded space
        non_recurrent_output = [nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=self.embed_dim),
            # nn.Tanh()
        )
            for _ in range(J)]
        self.non_recurrent_output = nn.ModuleList(non_recurrent_output)

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
        scale_embedding = self.non_recurrent_output[t](output)

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
