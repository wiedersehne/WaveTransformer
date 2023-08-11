import torch
import torch.nn as nn
from typing import Optional
from WaveLSTM.modules.Conv1dLSTM import Conv1dLSTM


class WaveletConv1dLSTM(nn.Module):
    r"""A wrapper around the Conv1dLSTM module.

    This class performs the non-recurrent projections from the reccurrent hidden state tensors of the last layer of
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
        J (int):
            The total number of resolutions to recurse over
    Kwargs:
        hidden_channels (int):
            The number of channels of the cell state. Default ``128``.
        kernel_size (int):
            Size of the convolutional kernel. Default ``3``.
        bias (bool):
            Whether to add the bias to convolutions. Default ``True``.
        proj_size (int):
            If ``>0``, will use ConvLSTM with hidden state projections with corresponding number of channels. Default ``0``.
        dropout (float):
            If non-zero, introduces a Dropout layer on the outputs of the Conv1dLSTM cell except the last layer,
             with dropout probability equal to dropout. Default: 0

    Inputs: input_tensor, (h_0, c_0)
        * **input_tensor**: tensor of shape :math:`(N, J, H_{in}, W)` for non-temporal input, otherwise `(N, time_steps, J, H_{in}, W)`
        * **h_0**: tensor of shape :math:`(\text{num\_layers}, N, H_{proj}, W)`
        * **c_0**: tensor of shape :math:`(\text{num\_layers}, N, H_{cell}, W)`

    Outputs: output, (h_n, c_n)
        * **output**: tensor of shape :math:`(N, D)` containing the resolution embedding for resolution j
        * **h_n**: List of L tensors of shape :math:`(N, H_{proj}, W)` containing the final hidden state
         tensor for resolution j
        * **c_n**: List of L tensor of shape :math:`(N, H_{cell}, W)` containing  the final cell state
         tensor for resolution j

        #TODO: This class should inherit Conv1dLSTM
    """
    def __str__(self):
        s = '\nWaveletLSTM'
        s += str(self.rnn)
        s += '\nNon-recurrent projection'
        s += f'\n\t to {self.resolution_embed_size} dimensional vector space'
        return s

    def __init__(self,
                 input_channels: int,
                 J: int,
                 hidden_channels: int = 128,
                 layers: int = 1,
                 proj_size: int = 0,
                 kernel_size: int = 7,
                 dropout: Optional[float] =None,
                 resolution_embed_size: int = 128,
                 ):
        """
        """
        super().__init__()

        self.input_channels = input_channels
        self.real_hidden_channels = proj_size if proj_size > 0 else hidden_channels
        self.resolution_embed_size = resolution_embed_size

        # LSTM
        self.rnn = Conv1dLSTM(input_channels=self.input_channels,
                              hidden_channels=hidden_channels,
                              kernel_size=kernel_size,
                              num_layers=layers,
                              proj_size=proj_size,
                              dropout=dropout,
                              )

        # Output layers to map to the embedded space
        non_recurrent_output = [nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=self.resolution_embed_size),
        )
            for _ in range(J)]
        self.non_recurrent_output = nn.ModuleList(non_recurrent_output)

    def forward(self, xj, hidden_state, j):
        """
        xj (torch.tensor):
            A resolution component, optionally temporal. Shape [batch size, time_steps, input_channels, signal_length]
            for temporal input, or [batch size, input_channels, signal_length]
        hidden_state (tuple),
            Hidden and cell state tensor. Shape: ([n layers, batch size, hid dim, signal length],  [n layers, batch size, hid dim, signal length])
        j (int)
            Resolution index [batch_size  (N, J, real_hidden_channels, Signal length)
        """

        xj = xj.view((xj.size(0), self.input_channels, -1))

        # Recurrent
        output, (h_next, c_next) = self.rnn(xj, hidden_state)
        # output: (N, time_steps, real_hidden_channels, signal length)

        # Output layer
        output = output[:, -1, :, :]                                # Take the last time_step
        scale_embedding = self.non_recurrent_output[j](output)

        return scale_embedding, (h_next, c_next)

