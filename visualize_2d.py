#!/usr/bin/env python3
# scripts/visualize_2d.py
# Compute a 2D loss landscape slice around a model.
import os, argparse, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
try:
    from models.mlp import SimpleMLP
except Exception:
    SimpleMLP = None

def get_data(dataset='mnist', subset=None, batch_size=512):
    if dataset=='mnist':
        transform = transforms.ToTensor()
        train = datasets.MNIST('.', train=True, download=True, transform=transform)
        if subset is not None:
            train = Subset(train, list(range(subset)))
        return DataLoader(train, batch_size=batch_size, shuffle=False)
    raise ValueError('Unsupported dataset in this script')

def flatten_params(params):
    return torch.cat([p.detach().view(-1) for p in params])

def set_params_from_flat(params, flat):
    idx=0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view_as(p))
        idx += n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mlp')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--subset', type=int, default=1024)
    parser.add_argument('--out', default='landscape.png')
    parser.add_argument('--grid', type=int, default=41)
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_data(args.dataset, subset=args.subset)
    xb, yb = next(iter(loader))
    xb, yb = xb.to(device), yb.to(device)

    if args.model=='mlp' and SimpleMLP is not None:
        model = SimpleMLP().to(device)
    else:
        raise ValueError('Only mlp supported in this script')

    params = [p for p in model.parameters() if p.requires_grad]
    theta0 = flatten_params(params).to(device)

    n = theta0.numel()
    dir1 = torch.randn(n, device=device)
    dir2 = torch.randn(n, device=device)
    dir1 = dir1 / (dir1.norm()+1e-12)
    dir2 = dir2 - (dir2.dot(dir1)/dir1.dot(dir1))*dir1
    dir2 = dir2 / (dir2.norm()+1e-12)

    grid = np.linspace(-args.scale, args.scale, args.grid)
    loss_map = np.zeros((args.grid, args.grid), dtype=float)
    loss_fn = nn.CrossEntropyLoss()

    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            theta = theta0 + a * dir1 + b * dir2
            set_params_from_flat(params, theta)
            model.eval()
            with torch.no_grad():
                logits = model(xb)
                loss = loss_fn(logits, yb).item()
            loss_map[j, i] = loss

    set_params_from_flat(params, theta0)
    plt.figure(figsize=(6,5))
    plt.contourf(grid, grid, loss_map, levels=60)
    plt.colorbar(label='loss')
    plt.xlabel('dir1 coefficient')
    plt.ylabel('dir2 coefficient')
    plt.title('2D loss landscape slice')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print('Saved', args.out)

if __name__=='__main__':
    main()
