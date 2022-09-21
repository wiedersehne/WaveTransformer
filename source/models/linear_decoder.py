import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearDecoder(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.decoder1 = nn.Linear(in_features=in_features, out_features=64)
        self.dec1 = nn.Linear(in_features=64, out_features=264)
        self.dec2 = nn.Linear(in_features=264, out_features=264)
        self.dec3 = nn.Linear(in_features=264, out_features=64)
        self.decoder2 = nn.Linear(in_features=64, out_features=out_features)

    def forward(self, x):
        x = F.relu(self.decoder1(x))
        x = F.relu(self.dec3(F.relu(self.dec2(F.relu(self.dec1(x))))))
        channel1 = torch.sigmoid(self.decoder2(x))
        return channel1
