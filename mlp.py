# models/mlp.py
import torch.nn as nn
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden=[256,128], num_classes=10):
        super().__init__()
        layers = []
        cur = input_dim
        for h in hidden:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            cur = h
        layers.append(nn.Linear(cur, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x)
