import torch
import torch.nn as nn

class MSCNN_BLOCK(nn.Module):
    """
    Input (Batch_size x Channel_size x Sequence_length)
    Output (Batch_size x class_number)
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernels):
        
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channel, out_channel, kernel, stride=kernel, padding = 256*(kernel - 1) // 2)
            for kernel in kernels
        ])
        self.bn = nn.BatchNorm1d(out_channel*len(kernels))
        self.activate = nn.ReLU()
        self.project = nn.Conv1d(out_channel*len(kernels), out_channel, 1, stride=1)

    def forward(self, x):
        out = [conv(x) for conv in self.convs]
        # print(x_n.shape, x_m.shape, x_w.shape)
        x = torch.cat(out, dim=1)
        x = self.activate(self.bn(x))
        x = self.project(x)

        return x, out
    

class MSCNN_NET(nn.Module):
    def __init__(self, in_channel, out_channel, kernels, num_layers):
        super().__init__()
        self.layers = num_layers
        self.mscnn_net = nn.ModuleList(
            [
                MSCNN_BLOCK(
                    in_channel,
                    out_channel,
                    kernels
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in range(self.layers):
            x, res_embeds = self.mscnn_net[layer](x)
        return x, res_embeds


class PREDICTION_HEAD(nn.Module):
    """
    Flatten by output.view(output.size(0), -1) 
    """
    def __init__(self, in_dimension, hidden_dimension, out_dimension, num_classes):
        super().__init__()

        self.clf_net = nn.Sequential(nn.Linear(in_features=in_dimension,
                                               out_features=hidden_dimension),
                                     nn.Tanh(),
                                     nn.Linear(in_features=hidden_dimension,
                                               out_features=out_dimension),
                                     nn.Tanh(),
                                     nn.LazyLinear(num_classes))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.clf_net(x)
        M = torch.unsqueeze(output, 1)
        return output, M