# Neural Network Loss Landscape Geometry & Optimization

**Repository**: `NN_loss_landsape_geometry_and_Optimization`

![CI](https://img.shields.io/badge/ci-passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This repo contains code and experiments to analyze neural network loss landscapes and their connection to optimization dynamics and generalization.
Key features added in this patched release:
- Hessian probing (power method, Lanczos, Hutchinson trace)
- Sharpness measurement (SAM-style)
- 2D loss surface visualization tools (`scripts/visualize_2d.py`)
- Streamlit dashboard for experiment plotting (`dash/app_streamlit.py`)
- CIFAR-10 support with a small ResNet (`models/resnet_small.py`) and a CIFAR training script (`experiments/train_and_probe_cifar.py`)

## Quickstart (CPU)

```bash
python -m venv venv
# activate venv (Windows PowerShell)
# venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run MNIST MLP experiment (small subset)
python -m experiments.train_and_probe --model mlp --dataset mnist --epochs 6 --subset 5000

# Run CIFAR-10 resnet experiment (small subset)
python -m experiments.train_and_probe_cifar --epochs 6 --subset 5000

# Visualize a 2D loss slice (example)
python scripts/visualize_2d.py --model mlp --dataset mnist --subset 1024 --out figures/landscape.png

# Launch dashboard (streamlit)
streamlit run dash/app_streamlit.py -- --results results
```

## Diagrams (ASCII placeholder)

```
 Training trajectory
    o-->o----->o
   /            \
  /   Loss surface \
 /                \
init           final
```

## Examples

- `experiments/train_and_probe.py` — MNIST MLP training + probes
- `experiments/train_and_probe_cifar.py` — CIFAR-10 ResNet training + probes
- `scripts/visualize_2d.py` — compute loss over a 2D grid in parameter space
- `dash/app_streamlit.py` — Streamlit dashboard to plot results and spectra

## License
MIT
