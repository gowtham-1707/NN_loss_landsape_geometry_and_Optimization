# experiments/train_and_probe_cifar.py
import argparse, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.resnet_small import ResNetSmall
from landscape.hessian import power_method_hv, hutchinson_trace
from landscape.sharpness import sharpness_l2

def get_data(subset=None, batch_size=128):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10('.', train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10('.', train=False, download=True, transform=transform_test)
    if subset is not None:
        train = Subset(train, list(range(subset)))
    return train, test

def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0,0,0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss_sum += loss_fn(logits,yb).item() * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds==yb).sum().item()
            total += xb.size(0)
    return loss_sum/total, correct/total

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    train_set, test_set = get_data(args.subset, batch_size=args.batch_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024)
    model = ResNetSmall().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)

    print('Recording init probes...')
    init_eigs = power_method_hv(lambda m,x,y: loss_fn(m(x),y), model, (xb,yb), k=3, iters=20)
    init_trace = hutchinson_trace(lambda m,x,y: loss_fn(m(x),y), model, (xb,yb), num_samples=10)
    init_sharp = sharpness_l2(model, lambda m,x,y: loss_fn(m(x),y), (xb,yb), eps=1e-3)
    print('Init eigenvalues:', init_eigs)
    print('Init trace:', init_trace)
    print('Init sharpness:', init_sharp)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f'Epoch {epoch}/{args.epochs} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}')

    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    print('Recording final probes...')
    final_eigs = power_method_hv(lambda m,x,y: loss_fn(m(x),y), model, (xb,yb), k=3, iters=20)
    final_trace = hutchinson_trace(lambda m,x,y: loss_fn(m(x),y), model, (xb,yb), num_samples=10)
    final_sharp = sharpness_l2(model, lambda m,x,y: loss_fn(m(x),y), (xb,yb), eps=1e-3)
    print('Final eigenvalues:', final_eigs)
    print('Final trace:', final_trace)
    print('Final sharpness:', final_sharp)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
