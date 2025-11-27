import torch
from landscape.hessian import hutchinson_trace, power_method_hv
from models.mlp import SimpleMLP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_small_batch():
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST('.', train=True, download=True, transform=transform)
    dl = DataLoader(train, batch_size=128, shuffle=True)
    xb, yb = next(iter(dl))
    return xb, yb

def test_hutchinson_trace_runs():
    xb, yb = get_small_batch()
    model = SimpleMLP()
    val = hutchinson_trace(lambda m,x,y: torch.nn.functional.cross_entropy(m(x), y), model, (xb,yb), num_samples=2)
    assert isinstance(val, float)

def test_power_method_runs():
    xb, yb = get_small_batch()
    model = SimpleMLP()
    eigs = power_method_hv(lambda m,x,y: torch.nn.functional.cross_entropy(m(x), y), model, (xb,yb), k=1, iters=5)
    assert isinstance(eigs, list)
