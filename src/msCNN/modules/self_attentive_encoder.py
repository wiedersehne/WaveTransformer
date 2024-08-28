from src.msCNN.modules.mscnn import MSCNN_NET
from abc import ABC
import torch
import torch.nn as nn
import pytorch_lightning as pl

class SelfAttentiveEncoder(pl.LightningModule, ABC):
    def __init__(self, in_channel, out_channel, kernels, num_layers, attention_unit, r_hops):
        super().__init__()
        self.encoder = MSCNN_NET(in_channel, out_channel, kernels, num_layers)
        # Attention
        self.ws1 = nn.LazyLinear(attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, r_hops, bias=False)
        self.init_weights(init_range=0.1)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, masked_inputs: torch.tensor, meta_data: dict):
        """
        x,
        """

        hidden, resolutions = self.encoder(masked_inputs)
        meta_data.update({'resolution_embeddings': [_hidden.detach().cpu().numpy() for _hidden in resolutions]})

        # hidden = torch.stack(hidden, dim=1)
        hidden = hidden.view(hidden.size()[0], len(resolutions), -1)    # [batch_size, num_kernels*n_hidden, seq_length] --> [batch_size, num_kernels, n_hidden]
        # print(hidden.shape)
        size = hidden.size()                                           # [batch_size, num_kernels, n_hidden]
        compressed_embeddings = hidden.view(-1, size[2])              # [batch_size * num_kernels, n_hidden]

        hbar = self.activation(self.ws1(compressed_embeddings))   # [batch_size * num_kernels, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)             # [batch_size, num_kernels, attention-hops]
        alphas = torch.transpose(alphas, 1, 2).contiguous()            # [batch_size, attention-hops, num_kernels]

        alphas = self.softmax(alphas.view(-1, size[1]))                # [batch_size * attention-hops, num_kernels]
        alphas = alphas.view(size[0], 1, size[1])    # [batch_size, attention-hops, num_kernels]
        meta_data.update({"attention": alphas.detach().cpu().numpy()})                        # A in Bengio's self-attention paper

        M = torch.bmm(alphas, hidden)                                 # [batch_size, attention-hops, n_hidden]

        return M, meta_data
