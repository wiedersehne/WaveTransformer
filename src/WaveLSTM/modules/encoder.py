from abc import ABC
import torch
import pytorch_lightning as pl
import numpy as np
import ptwt
import pywt
from WaveLSTM.modules.WaveConvLSTM import WaveletConv1dLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(pl.LightningModule, ABC):

    def masked_input(self, x, right=False, pool=True):
        """
        Get the input sequence from sequentially reconstructing partially masked wavelet coefficients
            right=True     :     get IWT(0, ..., alpha_{j}, ..., alpha_J) at input step j=1,2..
            right=False    :     get IWT(0, ..., alpha_j, 0, ...) at input step j=1,2..

        """
        assert x.dim() == 4

        width = self.masked_input_width if pool else x.size(3)

        masked_inputs = []
        for j in range(self.recursion_limit):
            masked_recon = torch.zeros((x.size(0), x.size(1), x.size(2), width), device=device)
            for c in range(x.shape[1]):
                for h in range(x.shape[2]):
                    full_bank = ptwt.wavedec(x[:, c, h, :], self.wavelet, mode='zero', level=self.max_level)
                    bank = full_bank[:self.recursion_limit] if pool else full_bank
                    masked_bank = [alpha_i if (i == j and not right) or (i >= j and right)
                                   else torch.zeros_like(alpha_i) for i, alpha_i in enumerate(bank)]
                    masked_recon[:, c, h, :] = ptwt.waverec(masked_bank, self.wavelet)

            masked_inputs.append(masked_recon)

        return masked_inputs

    def __init__(self,
                 seq_length, strands, chromosomes,
                 hidden_size=256, layers=1, proj_size=0, scale_embed_dim=128,
                 wavelet="haar",
                 recursion_limit=None,
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.wavelet = pywt.Wavelet(wavelet)
        self.chromosomes = chromosomes
        self.strands = strands

        # Wavelet input sequence
        # # Calculate recursion depth J
        self.max_level = pywt.dwt_max_level(seq_length, wavelet)
        self.recursion_limit = self.max_level if recursion_limit is None else np.min((recursion_limit, self.max_level))
        self.masked_input_width =  ptwt.wavedec(torch.zeros((1, seq_length)),
                                                self.wavelet, mode='zero',
                                                level=self.max_level)[self.recursion_limit - 1].shape[1] * 2

        # recurrent network
        self.wave_rnn = WaveletConv1dLSTM(out_features=seq_length, strands=strands, chromosomes=chromosomes,
                                          masked_input_width=self.masked_input_width,
                                          J=self.recursion_limit,
                                          hidden_size=hidden_size, layers=layers, proj_size=proj_size,
                                          scale_embed_dim=scale_embed_dim
                                          )

    def __str__(self):
        s = ''
        s += f'\nData'
        s += f'\n\t Data size (?, c={self.wave_rnn.C}, h={self.wave_rnn.H}, w={self.wave_rnn.W})'
        s += f'\nRecurrent network'
        s += f'\n\t Wavelet "{self.wavelet.name}", which has decomposition length {self.wavelet.dec_len}'
        s += f"\n\t J={self.recursion_limit}"
        s += f"\n\t Maximum resolution {1 / (self.wavelet.dec_len ** (self.recursion_limit - 1))} " \
             f"(as a fraction of each channel's width)"
        s += str(self.wave_rnn)
        return s

    def forward(self, x: torch.tensor):
        """
        x,                 features = [B, Strands, Chromosomes, Sequence Length]
        """
        assert x.dim() == 4

        # Partial fidelity truth, removing (zeroing) the contribution of coefficients further down the recurrence
        masked_inputs = self.masked_input(x)

        hidden_state = self.wave_rnn.init_states(x)                      # Initialise hidden and cell states
        hidden_embedding = []
        for j in range(self.recursion_limit):
            scale_embedding, hidden_state = self.wave_rnn(masked_inputs[j], hidden_state, j)
            hidden_embedding.append(scale_embedding)

        meta_data = {'masked_inputs': masked_inputs,
                     }

        return hidden_embedding, meta_data


