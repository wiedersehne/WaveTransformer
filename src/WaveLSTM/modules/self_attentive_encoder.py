from abc import ABC
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from WaveLSTM.modules.encoder import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttentiveEncoder(pl.LightningModule, ABC):
    """
    A class which wraps the wave-LSTM encoder in the self-attention mechanism of Bengio et al

    Refs: Bengio: https://arxiv.org/abs/1703.03130
          https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding/blob/master/models.py

    Args:
        TODO

    Kwargs:
        TODO

    Inputs: input_sequence, meta_data
        * **input_sequence**: list of tensors of shape :math:`(N, H_{in}, W)` for non-temporal input, otherwise `(N, time_steps, H_{in}, W)`
        * **meta_data**: dictionary of meta outputs

    Outputs: multi_resolution_embedding, meta_data
        * **resolution_embeddings**: list of tensors of shape :math:`(N, D)` containing the resolution embeddings $h_j$, j=1,...,J
        * **meta_data**: updated dictionary of meta outputs
    """
    def __init__(self, input_size, input_channels, r_hops=10, attention_unit=350, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.attention_hops = r_hops

        # Build encoder
        print(kwargs)
        self.encoder = Encoder(input_size,input_channels, **kwargs)

        # Attention
        self.ws1 = nn.LazyLinear(attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, r_hops, bias=False)
        self.init_weights(init_range=0.1)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def __str__(self):
        s = ''
        s += str(self.encoder)
        return s

    def forward(self, masked_inputs: torch.tensor, meta_data: dict):
        """
        x,
        """

        hidden, meta_data = self.encoder(masked_inputs, meta_data)
        meta_data.update({'resolution_embeddings': hidden})

        hidden = torch.stack(hidden, dim=1)
        size = hidden.size()                                           # [batch_size, num_multiscales, n_hidden]
        compressed_embeddings = hidden.view(-1, size[2])               # [batch_size * num_multiscales, n_hidden]

        hbar = self.activation(self.ws1(compressed_embeddings))   # [batch_size * num_multiscales, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)             # [batch_size, num_multiscales, attention-hops]
        alphas = torch.transpose(alphas, 1, 2).contiguous()            # [batch_size, attention-hops, num_multiscales]

        alphas = self.softmax(alphas.view(-1, size[1]))                # [batch_size * attention-hops, num_multiscales]
        alphas = alphas.view(size[0], self.attention_hops, size[1])    # [batch_size, attention-hops, num_multiscales]
        meta_data.update({"attention": alphas})                        # A in Bengio's self-attention paper

        M = torch.bmm(alphas, hidden)                                  # [batch_size, attention-hops, n_hidden]

        return M, meta_data
