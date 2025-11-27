# models/deep_linear.py
import torch
import torch.nn as nn
class DeepLinear(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False))
        self.layers = nn.ModuleList(layers)
    def forward(self, x):
        out = x
        for L in self.layers:
            out = L(out)
        return out
    def product(self):
        # return matrix product of weight matrices (as a torch Tensor)
        W = None
        for L in self.layers:
            if W is None:
                W = L.weight
            else:
                W = torch.matmul(L.weight, W)
        return W
