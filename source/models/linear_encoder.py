import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearEncoder(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.encoder1 = nn.Linear(in_features=in_features, out_features=64)
        self.enc1 = nn.Linear(in_features=64, out_features=264)
        self.enc2 = nn.Linear(in_features=264, out_features=264)
        self.enc3 = nn.Linear(in_features=264, out_features=64)
        self.encoder2 = nn.Linear(in_features=64, out_features=out_features)

    def forward(self, x):
        x = F.relu(self.encoder1(x))
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))

        x = self.encoder2(x)
        return x
