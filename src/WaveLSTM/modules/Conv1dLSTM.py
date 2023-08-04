# Modified from https://raw.githubusercontent.com/ndrplz/ConvLSTM_pytorch/master/convlstm.py
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Conv1dLSTMCell(nn.Module):

    def __init__(self, input_channels,
                 hidden_size: int = 128,
                 kernel_size: int = 3,
                 bias: bool = True,
                 proj_size: int = 0,
                 dropout_input: float = 0.,
                 dropout_proj: float = 0.
                 ):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_size: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether to add the bias.
        """

        super(Conv1dLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.proj_size = proj_size
        self.hidden_size = hidden_size
        self.real_hidden_size = proj_size if proj_size > 0 else hidden_size

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.drop_in = nn.Dropout(dropout_input)
        self.conv = nn.Conv1d(in_channels=self.input_channels + self.real_hidden_size,
                              out_channels=4 * self.hidden_size,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        if self.proj_size > 0:
            self.recurrent_drop = nn.Dropout(dropout_proj)
            self.recurrent_output = nn.Sequential(
                nn.Conv1d(in_channels=self.hidden_size,
                          out_channels=self.real_hidden_size,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                          bias=self.bias),
                nn.Tanh()
            )


    def forward(self, input_tensor, state):
        """
        input_tensor: (N, Channels, Width)
        """
        hidden, cell = state                                 # (N, hidden_size, width)
        input_tensor = self.drop_in(input_tensor)

        combined = torch.cat([input_tensor, hidden], dim=1)  # (N, Channels + real_hidden_size, width)
        combined_conv = self.conv(combined)                  # (N, 4 * hidden_size, width)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_size, dim=1)  # (N, hidden_size, width)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * cell + i * g                             #
        h_next = o * torch.tanh(c_next)

        if self.proj_size > 0:
            h_next = self.recurrent_output(self.recurrent_drop(h_next))                       # (N, proj_size, width)

        return h_next, c_next


class Conv1dLSTM(nn.Module):
    """
        Parameters:
            input_channels: Number of channels in input
            hidden_dim: Number of hidden channels
            kernel_size: Size of kernel in convolutions
            num_layers: Number of LSTM layers stacked on each other
            batch_first: Whether or not dimension 0 is the batch or not
            bias: Bias or no bias in Convolution
            return_all_layers: Return the list of computations for all layers
            Note: Will do same padding.

        Input:
            A tensor of size (B, T, C, L) or (T, B, C, L)
        Output:
         # output = (N, L, D=1 * H_out, Signal length)
        # hidden = (D=1 * num_layers, N, H_out, Signal length)
        # cell   = (D=1 * num_layers, N, H_out, Signal length)

            A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
                0 - layer_output_list is the list of lists of length T of each output
                1 - last_state_list is the list of last states
                        each element of the list is a tuple (h, c) for hidden state and memory
        Example:
            >> x = torch.rand((32, 10, 64, 128))
            >> convlstm = Conv1dLSTM(64, 16, 3, 1, True, True, False)
            >> _, last_states = convlstm(x)
            >> h = last_states[0][0]  # 0 for layer index, 0 for h index
        """
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 num_layers,
                 bias=True,
                 proj_size=0,
                 dropout_input: float = 0.,
                 dropout_hidden: float = 0.,
                 dropout_proj: float = 0.
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
        for i in range(self.num_layers):
            cell_list.append(Conv1dLSTMCell(input_channels=self.input_channels if i == 0 else self.real_hidden_channels,
                                            hidden_size=self.hidden_channels,
                                            kernel_size=self.kernel_size,
                                            bias=self.bias,
                                            proj_size=proj_size,
                                            dropout_input=dropout_input if i == 0 else dropout_hidden,
                                            dropout_proj=dropout_proj))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state):
        """

        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (b, t, c, l)
        hidden_state:
            last hidden/cell state
                (layers, N, H_out, signal length)

        Returns
        -------
        output,    Tensor of shape (L, N, D=1 * H_out, Signal length), or  (N, L, D=1 * H_out, Signal length)
                    when batch_first = True
                    containing the output features (h_t) from the last layer of the LSTM, for each t
        h_n,       Tensor of shape (D=1 * num_layers, N, H_out, Signal length)
                    containing the final hidden state for each element in the sequence
        c_n,       Tensor of shape (D=1 * num_layers, N, H_out, Signal length)
                    containing the final cell state for each element in the sequence
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

        output = torch.stack(output, dim=1)         # (N, L, H_out, Signal length)

        return output, (hidden, cell)

