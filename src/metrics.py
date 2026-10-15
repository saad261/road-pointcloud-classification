"""Evaluation helpers: run a model over a loader and compute classification metrics."""

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, jaccard_score,
                             precision_recall_fscore_support)


@torch.no_grad()
def predict_loader(model, loader, device, criterion=None):
    """Return (mean loss or None, true labels, predicted labels) for a whole loader."""
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        logits = model(points)
        if criterion is not None:
            total_loss += criterion(logits, labels).item() * points.size(0)
        n += labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy())
    mean_loss = total_loss / n if criterion is not None and n else None
    return mean_loss, np.array(y_true), np.array(y_pred)


def classification_report(y_true, y_pred, class_names):
    """Overall weighted metrics, per-class metrics and the confusion matrix."""
    labels = list(range(len(class_names)))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    pc_p, pc_r, pc_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(p),
        "weighted_recall": float(r),
        "weighted_f1": float(f1),
        "weighted_iou": float(jaccard_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            name: {"precision": float(pc_p[i]), "recall": float(pc_r[i]),
                   "f1": float(pc_f1[i]), "support": int(support[i])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def format_confusion_matrix(cm, class_names):
    width = max(len(c) for c in class_names) + 2
    header = "True \\ Pred".ljust(width) + "".join(c.rjust(width) for c in class_names)
    rows = [header, "-" * len(header)]
    for name, row in zip(class_names, cm):
        rows.append(name.ljust(width) + "".join(str(v).rjust(width) for v in row))
    return "\n".join(rows)
