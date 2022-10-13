import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearDecoder(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=32):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=out_features)
        )

    def forward(self, z):
        return self.decoder(z)


class WrappedLinearDecoder(nn.Module):
    def __init__(self, in_features, out_features, strands, hidden_features=32):
        super().__init__()
        self.out_features = out_features
        self.strands = strands

        self.net = LinearDecoder(in_features,
                                 out_features * strands,
                                 hidden_features=hidden_features)

    def forward(self, z):
        return torch.reshape(self.net(z), (z.size(0), self.strands, self.out_features))
