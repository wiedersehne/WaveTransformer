import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearDecoder(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()

        self.dec1 = nn.Linear(in_features=in_features, out_features=64)
        self.dec2 = nn.Linear(in_features=64, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=out_features)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.dec1(x)))
        x = F.relu(self.bn2(self.dec2(x)))
        x = self.dec3(x)

        return x
