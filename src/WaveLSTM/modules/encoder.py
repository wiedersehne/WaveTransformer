from abc import ABC
import torch
import pytorch_lightning as pl
import numpy as np
# import ptwt
import pywt
from torch import nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.WaveLSTM.modules.WaveConvLSTM import WaveletConv1dLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(pl.LightningModule, ABC):
    r"""
    Recurrently encode the input sequence

    Args:
        TODO

    Kwargs:
        TODO

    Inputs: input_sequence, meta_data
        * **input_sequence**: list of tensors of shape :math:`(N, H_{in}, W)` for non-temporal input, otherwise `(N, time_steps, H_{in}, W)`
        * **meta_data**: dictionary of meta outputs

    Outputs: resolution_embedding, meta_data
        * **resolution_embeddings**: list of tensors of shape :math:`(N, D)` containing the resolution embeddings $h_j$, j=1,...,J
        * **meta_data**: updated dictionary of meta outputs

    """

    def __init__(self,
                 input_size,
                 input_channels,
                 J,
                 D=128,
                 hidden_channels=256,
                 layers=1,
                 proj_size=0,
                 kernel_size=7,
                 wavelet="haar",
                 dropout=0,
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.wavelet = pywt.Wavelet(wavelet)
        self.input_channels = input_channels
        self.input_size = input_size
        self.J = J

        self.real_hidden_channels = proj_size if proj_size > 0 else hidden_channels
        self.hidden_channels = hidden_channels
        self.layers = layers

        # recurrent network
        self.wave_rnn = WaveletConv1dLSTM(input_channels=input_channels,
                                          hidden_channels=hidden_channels,
                                          J=self.J,
                                          layers=layers,
                                          proj_size=proj_size,
                                          kernel_size=kernel_size,
                                          dropout=dropout,
                                          resolution_embed_size=D,
                                          )

    def __str__(self):
        s = ''
        s += f'\nData'
        s += f'\n\t Data size (?, c={self.wave_rnn.C}, h={self.wave_rnn.H}, w={self.wave_rnn.W})'
        s += f'\nRecurrent network'
        s += f'\n\t Wavelet "{self.wavelet.name}", which has decomposition length {self.wavelet.dec_len}'
        s += f"\n\t J={self.J}"
        s += f"\n\t Maximum resolution {1 / (self.wavelet.dec_len ** (self.J - 1))} " \
             f"(as a fraction of each channel's width)"
        s += str(self.wave_rnn)
        return s

    def forward(self, input_sequence: torch.tensor, meta_data: dict):
        """
        x,                 features = [B, Strands, Chromosomes, Sequence Length]
        """
        assert len(input_sequence) == self.J
        assert np.all([xj.shape[1] == self.input_channels for xj in input_sequence]), \
            f"{[xj.shape[1] for xj in input_sequence]} != {[self.input_channels for _ in input_sequence]}"
        assert np.all([xj.shape[2] == self.input_size for xj in input_sequence]), \
            f"{[xj.shape[2] for xj in input_sequence]} != {[self.input_size for _ in input_sequence]}"

        # Initialise hidden and cell states
        hidden_state = self.init_states(input_sequence[0].size(0))

        # Loop over multiscales
        resolution_embeddings = []
        for j in range(self.J):
            xj = input_sequence[j]
            scale_embedding, hidden_state = self.wave_rnn(xj, hidden_state, j)
            resolution_embeddings.append(scale_embedding)

        return resolution_embeddings, meta_data

    def init_states(self, batch_size):
        """
        Initial hidden and cell tensor states. Initialised separately for each layer.
        """
        return (torch.zeros((self.layers,
                             batch_size,
                             self.real_hidden_channels,
                             self.input_size), device=device),
                torch.zeros((self.layers,
                             batch_size,
                             self.hidden_channels,
                             self.input_size), device=device))

