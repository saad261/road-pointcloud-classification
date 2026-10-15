"""Rebuild the result figures in docs/figures/ from results/training_log.txt.

    python scripts/make_figures.py
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LOG = os.path.join(ROOT, "results", "training_log.txt")
OUT = os.path.join(ROOT, "docs", "figures")
CLASSES = ["2lanes", "3lanes", "crossing", "split4lanes", "split6lanes", "transition"]

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
})


def parse_log(text):
    epochs = [tuple(map(float, m)) for m in re.findall(
        r"Epoch (\d+)/\d+: Train Loss: ([\d.]+), Train Acc: ([\d.]+) \| Val Loss: ([\d.]+), Val Acc: ([\d.]+)", text)]
    block = text.split("Confusion Matrix:")[1].split("Confusion Matrix (Rows")[0]
    cm = np.array([[int(v) for v in row.split()] for row in re.findall(r"\[+([\d\s]+)\]", block)])
    return np.array(epochs), cm


def training_curves(ep):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (tr, va, title) in zip(axes, [(1, 3, "Loss"), (2, 4, "Accuracy")]):
        ax.plot(ep[:, 0], ep[:, tr], color=BLUE, lw=2, marker="o", ms=5, label="Train")
        ax.plot(ep[:, 0], ep[:, va], color=ORANGE, lw=2, marker="o", ms=5, label="Validation")
        ax.set_title(title, color=INK, loc="left", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.axvline(66, color=INK2, ls=":", lw=1)
        lo, hi = ax.get_ylim()
        ax.text(65, lo + 0.5 * (hi - lo), "early stop\n(epoch 66) ", color=INK2, ha="right", va="center", fontsize=9)
        # direct labels at line ends
        ax.annotate("Train", (ep[-1, 0], ep[-1, tr]), xytext=(6, 0), textcoords="offset points", color=INK, va="center", fontsize=9)
        ax.annotate("Val", (ep[-1, 0], ep[-1, va]), xytext=(6, 0), textcoords="offset points", color=INK, va="center", fontsize=9)
        ax.set_xlim(5, 72)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Training history (logged every 10 epochs)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_curves.png"))
    plt.close(fig)


def confusion(cm):
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Test-set confusion matrix (n = {cm.sum()})", color=INK, loc="left", fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confusion_matrix.png"))
    plt.close(fig)


def per_class(cm):
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    recall = tp / cm.sum(1)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    x, w = np.arange(len(CLASSES)), 0.26
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for k, (vals, color, name) in enumerate([(precision, BLUE, "Precision"), (recall, ORANGE, "Recall"), (f1, AQUA, "F1")]):
        ax.bar(x + (k - 1) * w, vals, width=w - 0.03, color=color, label=name)
    for i, v in enumerate(f1):
        ax.text(i + w, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{c}\n(n={int(s)})" for c, s in zip(CLASSES, cm.sum(1))])
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.set_title("Per-class test metrics (F1 labelled)", color=INK, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "per_class_metrics.png"))
    plt.close(fig)
    return precision, recall, f1


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    epochs, cm = parse_log(open(LOG).read())
    training_curves(epochs)
    confusion(cm)
    p, r, f = per_class(cm)
    print("class        prec   rec    f1     n")
    for c, a, b, d, n in zip(CLASSES, p, r, f, cm.sum(1)):
        print(f"{c:<12} {a:.3f}  {b:.3f}  {d:.3f}  {n}")
    print("accuracy", np.trace(cm) / cm.sum())
