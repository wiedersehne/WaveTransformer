import torch
import torch.nn as nn
import pywt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StackedWavelet1D(nn.Module):

    @property
    def wavelet(self):
        return pywt.Wavelet("haar")

    @property
    def bank_shape(self):
        # List of  sequence lengths along filter bank
        return self._bank_shape

    @bank_shape.setter
    def bank_shape(self, batch):
        bank = pywt.wavedec(batch, self.wavelet)
        self._bank_shape = [tmp.shape[-1] for tmp in bank]

    # def multilevel_idwt(self, concat_filter_bank):
    #     """Multi-level inverse discrete wavelet transform
    #     """
    #     filter_bank = torch.split(concat_filter_bank, split_size_or_sections=self.bank_shape, dim=-1)
    #     # print([tmp.shape for tmp in filter_bank])
    #     feature_recon = pywt.waverec([tmp.detach().numpy() for tmp in filter_bank], 'haar')
    #     return torch.tensor(feature_recon).float().to(device)

    def __init__(self, in_features, out_features, strands, chromosomes, hidden_features=32):
        super().__init__()
        self.out_features = out_features
        self.strands = strands
        self.chromosomes = chromosomes

        # Get the dimensions of the filter bank
        self.bank_shape = torch.ones((2, strands, chromosomes, out_features))

        # FC Network from the latent samples to the concat levels of the filter bank
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features,
                      out_features=self.chromosomes * self.strands * int(np.sum(self.bank_shape)))
        )

    def forward(self, z):
        filter_bank = self.net(z).reshape(z.shape[0], self.strands, self.chromosomes, int(np.sum(self.bank_shape)))
        # y_recon = self.multilevel_idwt(filter_bank)
        return filter_bank
