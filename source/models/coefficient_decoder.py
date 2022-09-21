import torch
import torch.nn as nn
import torch.nn.functional as F


class CoefficientDecoder(nn.Module):
    def __init__(self, in_features, bases):
        super().__init__()
        self.bases = bases

        self.dec1 = nn.Linear(in_features=in_features, out_features=64)
        self.dec2 = nn.Linear(in_features=64, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=self.bases.shape[0])

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):

        x = F.relu(self.bn1(self.dec1(x)))
        x = F.relu(self.bn2(self.dec2(x)))
        z = self.dec3(x)

        return torch.matmul(z, torch.Tensor(self.bases).to(z.device))
