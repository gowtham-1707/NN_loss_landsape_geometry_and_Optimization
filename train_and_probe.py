import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from landscape.hessian import power_method_hv, hutchinson_trace
from landscape.sharpness import sharpness_l2
from models.mlp import SimpleMLP


def get_data(dataset, subset_size=None):
    if dataset == "mnist":
        transform = transforms.ToTensor()
        train = datasets.MNIST(".", train=True, download=True, transform=transform)
        test = datasets.MNIST(".", train=False, download=True, transform=transform)

        if subset_size is not None:
            train = Subset(train, list(range(subset_size)))

        return train, test

    raise ValueError("Unsupported dataset")


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

    return loss_sum / total, correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load data
    train_set, test_set = get_data(args.dataset, args.subset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024)

    # Model
    model = SimpleMLP().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Initial probes
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)

    print("Recording init probes...")
    init_eigs = power_method_hv(lambda m, x, y: loss_fn(m(x), y),
                                model, (xb, yb), k=3, iters=20)
    init_trace = hutchinson_trace(lambda m, x, y: loss_fn(m(x), y),
                                  model, (xb, yb), num_samples=10)
    init_sharp = sharpness_l2(model, lambda m, x, y: loss_fn(m(x), y),
                              (xb, yb), eps=1e-3)

    print("Init eigenvalues:", init_eigs)
    print("Init trace:", init_trace)
    print("Init sharpness:", init_sharp)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    # Final probes
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)

    print("Recording final probes...")
    final_eigs = power_method_hv(lambda m, x, y: loss_fn(m(x), y),
                                 model, (xb, yb), k=3, iters=20)
    final_trace = hutchinson_trace(lambda m, x, y: loss_fn(m(x), y),
                                   model, (xb, yb), num_samples=10)
    final_sharp = sharpness_l2(model, lambda m, x, y: loss_fn(m(x), y),
                               (xb, yb), eps=1e-3)

    print("Final eigenvalues:", final_eigs)
    print("Final trace:", final_trace)
    print("Final sharpness:", final_sharp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
