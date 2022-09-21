import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # TODO: Shouldn't be needed with lightning


class CoefficientDecoder(nn.Module):
    def __init__(self, in_features, seq_length, kernels):
        super().__init__()
        raise NotImplementedError    # deprecated TODO: fix
        # unzip all bases from kernels
        bases = np.stack([[base[1] for base in class_kernels] for class_kernels in kernels])
        self.bases = torch.Tensor(bases.reshape((-1, seq_length))).to(device)

        self.decoder1 = nn.Linear(in_features=in_features, out_features=64)
        self.dec1 = nn.Linear(in_features=64, out_features=64)
        self.dec2 = nn.Linear(in_features=64, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=64)
        self.decoder2 = nn.Linear(in_features=64, out_features=self.bases.shape[0])

    def forward(self, x):

        x = F.relu(self.decoder1(x))

        x = self.dec1(self.dec2(self.dec3(x)))

        z = self.decoder2(x)
        channel1 = torch.zeros((z.shape[0], self.bases.shape[1])).to(device)
        for idx_base in range(self.bases.shape[0]):
            for idx_n in range(channel1.shape[0]):
                channel1[idx_n, :] += z[idx_n, idx_base] * self.bases[idx_base, :]


        #channel1 = torch.matmul(channel1, self.bases)

        #channel2 = F.relu(self.decoder2_2(x))
        #channel2 = torch.matmul(channel2, self.bases)

        #reconstruction = torch.stack([channel1, channel1], dim=2)
        #reconstruction = reconstruction.unsqueeze(2)
        return channel1

