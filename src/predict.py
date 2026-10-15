"""Classify one or more road block point clouds (.npy) with the trained SimplePointNet.

Examples:
    python src/predict.py path/to/block.npy
    python src/predict.py path/to/folder_of_npy_files --top-k 3

Each .npy file must have the same raw column layout as the training data
(22 features per point); the 11 used features are selected automatically.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from config import SELECTED_FEATURE_INDICES
from dataset import collate_pad
from model import SimplePointNet

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "..", "models", "simplepointnet_best.pth")
DEFAULT_STATS = os.path.join(HERE, "..", "models", "normalization_stats.json")


def load_cloud(path, fmin, fmax):
    pts = np.load(path).astype(np.float32)[:, SELECTED_FEATURE_INDICES]
    rng = np.where(fmax - fmin == 0, 1e-6, fmax - fmin)
    return torch.from_numpy((pts - fmin) / rng).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".npy files or folders containing .npy files")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--stats", default=DEFAULT_STATS)
    ap.add_argument("--top-k", type=int, default=1)
    args = ap.parse_args()

    with open(args.stats) as f:
        stats = json.load(f)
    fmin, fmax = np.array(stats["feature_min"], np.float32), np.array(stats["feature_max"], np.float32)
    class_names, max_points = stats["class_names"], stats["max_points"]

    model = SimplePointNet(len(fmin), len(class_names))
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()

    files = []
    for p in args.inputs:
        files += sorted(glob.glob(os.path.join(p, "**", "*.npy"), recursive=True)) if os.path.isdir(p) else [p]

    with torch.no_grad():
        for path in files:
            # Pad to the same size used during training so results match training conditions
            x, _ = collate_pad([(load_cloud(path, fmin, fmax), torch.tensor(0))], max_points)
            probs = torch.softmax(model(x), dim=1)[0]
            top = torch.topk(probs, k=min(args.top_k, len(class_names)))
            guesses = ", ".join(f"{class_names[i]} ({p:.1%})" for p, i in zip(top.values.tolist(), top.indices.tolist()))
            print(f"{path}: {guesses}")


if __name__ == "__main__":
    main()
