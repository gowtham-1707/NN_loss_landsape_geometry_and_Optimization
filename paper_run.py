#!/usr/bin/env python3
# experiments/paper_run.py -- runs a set of canonical experiments and saves figures used in the paper
import os, json
from experiments.train_and_probe import main as train_main
# Simple wrapper to run small experiments and save outputs in results/paper/
os.makedirs('results/paper', exist_ok=True)
# run small mnist probe
train_main(argparse.Namespace(model='mlp', dataset='mnist', epochs=6, batch_size=256, lr=0.1, subset=5000))
print('Paper experiments completed (demo).')
