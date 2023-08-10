import torch.nn as nn
import torch
class Conv1dLSTMCell(nn.Module):

    def __init__(self,
                 input_channels : int,
                 hidden_size: int = 128,
                 kernel_size: int = 7,
                 bias: bool = True,
                 proj_size: int = 0,
                 dropout: float = 0.,
                 ):
        r"""
        A convolutional LSTM cell, with optional hidden state projection.

        For an element :math:`\hat{X}_j` from a sequence of channelised signals, this cell computes
        the following set of convolutional operations

        .. math::
            \begin{array}{ll} \\
                i_j &= \sigma(W_{ii} * \hat{X}_j + b_{ii} + W_{hi} * \H_{j-1} + b_{hi}) \\
                f_j &= \sigma(W_{if} * \hat{X}_j + b_{if} + W_{hf} * \H_{j-1} + b_{hf}) \\
                g_j &= \tanh(W_{ig} * \hat{X}_j + b_{ig} + W_{hg} * \H_{j-1} + b_{hg}) \\
                o_j &= \sigma(W_{io} * \hat{X}_j + b_{io} + W_{ho} * \H_{j-1} + b_{ho}) \\
                C_j &= f_j \odot C_{j-1} + i_j \odot g_j \\
                H_j &= o_j \odot \tanh(C_j),
            \end{array}

        where :math:`H_j` is the hidden state tensor at resolution `j`, :math:`C_j` is the cell
        state tensor at resolution `j`, :math:`\hat{X}_j` is the input at resolution `j`, :math:`H_{j-1}`
        is the hidden state tensor of the layer at resolution `j-1` or the initial hidden
        state tensor at resolution `0`, and :math:`i_j`, :math:`f_j`, :math:`g_j`,
        :math:`o_j` are the input, forget, cell, and output gates, respectively.
        :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

        If ``proj_size > 0`` is specified, Conv1dLSTM with projections will be used. This changes
        the LSTM cell in the following way. First, the number of channels of :math:`H_j` will be changed from
        ``hidden_size`` to ``proj_size`` (dimensions of :math:`W_{hi}` will be changed accordingly).
        Second, the output hidden state tensor of each layer will be multiplied by a learnable projection
        matrix: :math: `H_j = W_{oh} * H_j`. Note that as a consequence of this, the output
        of LSTM network will be of different shape as well.

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
                If ``>0``, will use ConvLSTM with hidden state projections with corresponding number of channels.
                 Default ``0``.
            dropout (float):
                If non-zero, introduces a Dropout layer on the outputs of the Conv1dLSTM cell,
                 with dropout probability equal to dropout. Default: 0
        """

        super(Conv1dLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.proj_size = proj_size
        self.hidden_size = hidden_size
        self.real_hidden_size = proj_size if proj_size > 0 else hidden_size

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(in_channels=self.input_channels + self.real_hidden_size,
                              out_channels=4 * self.hidden_size,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        if self.proj_size > 0:
            self.recurrent_output = nn.Sequential(
                nn.Conv1d(in_channels=self.hidden_size,
                          out_channels=self.real_hidden_size,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                          bias=self.bias),
                nn.Tanh()
            )


    def forward(self,
                input_tensor,
                state: tuple):
        """
        input_tensor: (N, Channels, Width)
        """
        hidden, cell = state                                 # (N, hidden_size, width)
        # input_tensor = self.dropout(input_tensor)

        combined = torch.cat([input_tensor, hidden], dim=1)  # (N, Channels + real_hidden_size, width)
        combined_conv = self.conv(combined)                  # (N, 4 * hidden_size, width)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_size, dim=1)  # (N, hidden_size, width)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * cell + i * g
        h_next = o * torch.tanh(c_next)

        if self.proj_size > 0:
            h_next = self.recurrent_output(h_next)                       # (N, proj_size, width)

        h_next = self.dropout(h_next)
        c_next = self.dropout(c_next)

        return h_next, c_next


class Conv1dLSTM(nn.Module):
    r"""Applies a multi-layer convolutional long short-term memory (LSTM) RNN to an input sequence.

    For each element in the input sequence :math:`\{\hat{X}_j\}_{j=1}^J`, each layer  computes
    the following set of convolutional operations

    .. math::
        \begin{array}{ll} \\
            i_j &= \sigma(W_{ii} * \hat{X}_j + b_{ii} + W_{hi} * \H_{j-1} + b_{hi}) \\
            f_j &= \sigma(W_{if} * \hat{X}_j + b_{if} + W_{hf} * \H_{j-1} + b_{hf}) \\
            g_j &= \tanh(W_{ig} * \hat{X}_j + b_{ig} + W_{hg} * \H_{j-1} + b_{hg}) \\
            o_j &= \sigma(W_{io} * \hat{X}_j + b_{io} + W_{ho} * \H_{j-1} + b_{ho}) \\
            C_j &= f_j \odot C_{j-1} + i_j \odot g_j \\
            H_j &= o_j \odot \tanh(C_j),
        \end{array}

    where :math:`H_j` is the hidden state tensor at resolution `j`, :math:`C_j` is the cell
    state tensor at resolution `j`, :math:`\hat{X}_j` is the input at resolution `j`, :math:`H_{j-1}`
    is the hidden state tensor of the layer at resolution `j-1` or the initial hidden
    state tensor at resolution `0`, and :math:`i_j`, :math:`f_j`, :math:`g_j`,
    :math:`o_j` are the input, forget, cell, and output gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    In a multilayer LSTM, the input :math:`\hat{X}^{(l)}_j` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state tensor :math:`H^{(l-1)}_j` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_j` where each :math:`\delta^{(l-1)}_j` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    If ``proj_size > 0`` is specified, Conv1dLSTM with projections will be used. This changes
    the LSTM cell in the following way. First, the number of channels of :math:`H_j` will be changed from
    ``hidden_size`` to ``proj_size`` (dimensions of :math:`W_{hi}` will be changed accordingly).
    Second, the output hidden state tensor of each layer will be multiplied by a learnable projection
    matrix: :math: `H_j = W_{oh} * H_j`. Note that as a consequence of this, the output
    of LSTM network will be of different shape as well.

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

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                J ={} & \text{sequence length} \\
                W ={} & \text{input\_size (input signal length)} \\
                H_{in} ={} & \text{input\_channels} \\
                H_{cell} ={} & \text{hidden\_size} \\
                H_{proj} ={} & \text{proj\_size if } \text{proj\_size}>0 \text{ otherwise hidden\_size} \\
            \end{aligned}

    .. note::
        Bidirectional ConvLSTM is not implemented

    Examples::

        >>> x = torch.rand((32, 10, 64, 128))
        >>> convlstm = Conv1dLSTM(64, 16, 3, 1, True, True, False)
        >>> _, last_states = convlstm(x)
        >>> h = last_states[0][0]  # 0 for layer index, 0 for h index
        """
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size :int = 7,
                 num_layers: int = 1,
                 bias : bool =True,
                 proj_size:int =0,
                 dropout: float = 0.,
                 ):

        super(Conv1dLSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.real_hidden_channels = hidden_channels if proj_size == 0 else proj_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias

        cell_list = []
        hidden_to_input_list = []
        for l in range(self.num_layers):
            cell_list.append(Conv1dLSTMCell(input_channels=self.input_channels if l == 0 else self.real_hidden_channels,
                                            hidden_size=self.hidden_channels,
                                            kernel_size=self.kernel_size,
                                            bias=self.bias,
                                            proj_size=proj_size,
                                            dropout=dropout if l<self.num_layers-1 else 0,
                                            ))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state):
        """
        """

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(1)                 # (b, t, c, signal_length)
        b, seq_len, _, sig_len = input_tensor.size()

        hidden, cell = hidden_state                               # (layers, N, H_out, signal length)

        output = []
        for xt in input_tensor.split(1, dim=1):
            xt = xt.squeeze(1)

            hidden_last, cell_last = [], []
            for layer_idx in range(self.num_layers):
                h_next, c_next = self.cell_list[layer_idx](input_tensor=xt, state=[hidden[layer_idx], cell[layer_idx]])
                xt = h_next

                hidden_last.append(h_next)
                cell_last.append(c_next)
            hidden, cell = hidden_last, cell_last          # (num_layers, N, H_out, Signal length)

            # after iterating over layers
            output.append(hidden[-1])

        output = torch.stack(output, dim=1)         # (N, J, H_proj, Signal length)

        return output, (hidden, cell)

