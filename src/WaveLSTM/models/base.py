import warnings

import torch
import numpy as np
import ptwt
import pywt
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WaveletBase(nn.Module):

    @property
    def normalize_stats(self):
        raise NotImplementedError
        # return self.mu_features, self.std_features

    @normalize_stats.setter
    def normalize_stats(self, stats):
        raise NotImplementedError
        # for stat in stats:
        #     assert stat.dim() == 2
        #     assert stat.size(0) == self.input_channels
        #     assert stat.size(1) == self.input_size
        # self.mu_features = stats[0].to(device)
        # self.std_features = stats[1].to(device)

    def scale(self, features):
        raise NotImplementedError
        # assert features.dim() == 3, features.dim()
        # if (self.mu_features is not None) and (self.std_features is not None):
        #     batch_mu = torch.concat([self.mu_features[None, :, :] for _ in range(features.size(0))], 0)
        #     batch_std = torch.concat([self.std_features[None, :, :] for _ in range(features.size(0))], 0)
        #     features -= batch_mu
        #     features /= batch_std
        # return features

    def unscale(self, normalized_features):
        raise NotImplementedError
        # assert normalized_features.dim() == 3, normalized_features.dim()
        # if (self.mu_features is not None) and (self.std_features is not None):
        #     batch_mu = torch.concat([self.mu_features[None, :, :] for _ in range(normalized_features.size(0))], 0)
        #     batch_std = torch.concat([self.std_features[None, :, :] for _ in range(normalized_features.size(0))], 0)
        #     normalized_features *= batch_std
        #     normalized_features += batch_mu
        # return normalized_features

    def sequence_mask(self, x, pool_targets=False):
        """
        Get the input sequence from sequentially reconstructing partially masked wavelet coefficients.
        Down-samples LSTM inputs (by removing j>J resolutions)

                input:         get IWT(0, ..., alpha_j, 0, ...)
                target:        get IWT(alpha_1, ..., alpha_{J}) if pool_target is True
                                    else IWT(alpha_1, ..., alpha_{J}, 0, ...)

        j>J are removed from the filter bank, effectively applying an average pooling operation.

        return: Down-sampled source-seShowparated input signals, and
                Filtered and down-sampled target signal
        """
        assert x.dim() == 3, x.size()
        pool_inputs = False

        # Masked inputs
        masked_inputs = []
        for j in range(self.J):
            X_j = torch.zeros((x.size(0), x.size(1), self.masked_width if pool_inputs else x.size(2)), device=device)
            for c in range(x.shape[1]):
                # Filter bank over channel sequence                
                full_bank = ptwt.wavedec(x[:, c, :], self.wavelet, mode='zero', level=self.max_detail_spaces)
                # Mask (see doc string) wavelet coefficients alpha_j
                masked_bank = [alpha_i if i == j else torch.zeros_like(alpha_i) for i, alpha_i in enumerate(full_bank)]
                # Remove j>J, and in doing this the reconstruction below will be average pooled
                trunc_bank = masked_bank[:self.J]
                # Reconstruct source-separated input
                X_j[:, c, :] = ptwt.waverec(trunc_bank if pool_inputs else masked_bank, self.wavelet)
            masked_inputs.append(X_j)

        # Targets (sequence of IWT(alpha_1, ...., alpha_j) for j={1,2,3...}).
        # The last of sequence is used as target, the rest are used for visualisation
        masked_targets = []
        for j in range(self.J):
            if pool_targets:
                masked_target = torch.zeros((x.size(0), x.size(1), self.masked_width), device=device)
            else:
                masked_target = torch.zeros_like(x, device=device)
            for c in range(x.shape[1]):
                # Filter bank over channel sequence
                full_bank = ptwt.wavedec(x[:, c, :], self.wavelet, mode='zero', level=self.max_detail_spaces)
                # Mask (see doc string) wavelet coefficients alpha_j
                masked_bank = [alpha_i if i <= j else torch.zeros_like(alpha_i) for i, alpha_i in enumerate(full_bank)]
                # Remove j>J, and in doing this the reconstruction below will be average pooled
                if pool_targets:
                    masked_bank = masked_bank[:self.J]
                # Reconstruct truncated target signal
                masked_target[:, c, :] = ptwt.waverec(masked_bank, self.wavelet)

            # Normalise scale if pooled
            if pool_targets:
                masked_target/= np.log2(x.size(2)/self.masked_width)

            masked_targets.append(masked_target)

        return masked_inputs, masked_targets

    def __init__(self, input_size, input_channels, recursion_limit=None, wavelet="haar", batch_norm=True):
        super().__init__()

        self.wavelet = wavelet
        self.input_size = input_size

        # Wavelet input sequence
        # # Calculate recursion depth J
        self.max_detail_spaces = pywt.dwt_max_level(self.input_size, wavelet)
        self.J = self.max_detail_spaces + 1 if recursion_limit is None \
            else np.min((recursion_limit, self.max_detail_spaces + 1))

        # Get the vector space dimension of each alpha_j
        _bank = pywt.wavedec(np.zeros((1, 1, self.input_size)), self.wavelet, level=self.max_detail_spaces)
        self.alpha_lengths = [b.shape[-1] for b in _bank]
        self.masked_width = pywt.waverec(_bank[:self.J], self.wavelet).shape[-1]
        
        if batch_norm:
            self.batch_norms = [nn.BatchNorm1d(num_features=input_channels, device=device) for _ in range(self.J)]
        else:
            self.batch_norms = None

    def forward(self, input, **kwargs):
        masked_inputs, masked_targets = self.sequence_mask(input, **kwargs)
        # print(masked_inputs.shape)
        if self.batch_norms is not None:
            masked_inputs = [self.batch_norms[j](masked_inputs[j]) for j in range(self.J)]

        return masked_inputs, masked_targets