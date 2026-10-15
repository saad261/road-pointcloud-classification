"""Train SimplePointNet on road block point clouds and evaluate it on a held-out test split.

Example:
    python src/train.py --data-root data/dataset --out-dir runs/exp1

The defaults reproduce the setup of the reported run (lr 7e-5, weight decay 5e-4,
batch size 8, dropout 0.4, up to 200 epochs, early stopping after 30 epochs
without validation-loss improvement, 80/15/5 split, seed 42).
"""

import argparse
import json
import os
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from config import CLASS_NAMES, DEFAULT_HPARAMS, FEATURE_NAMES, SEED, SPLIT_RATIOS
from dataset import (RoadPointCloudDataset, SubsetWithTransform, collate_pad,
                     compute_min_max, find_max_points, set_seed, train_augmentation)
from metrics import classification_report, format_confusion_matrix, predict_loader
from model import SimplePointNet, count_parameters


def parse_args():
    hp = DEFAULT_HPARAMS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder with one sub-folder of .npy files per class")
    ap.add_argument("--out-dir", default="runs/latest", help="where weights, stats and metrics are written")
    ap.add_argument("--lr", type=float, default=hp["learning_rate"])
    ap.add_argument("--weight-decay", type=float, default=hp["weight_decay"])
    ap.add_argument("--batch-size", type=int, default=hp["batch_size"])
    ap.add_argument("--dropout", type=float, default=hp["dropout_rate"])
    ap.add_argument("--epochs", type=int, default=hp["num_epochs"])
    ap.add_argument("--patience", type=int, default=hp["early_stop_patience"], help="early-stopping patience (epochs)")
    ap.add_argument("--augment", action="store_true", help="enable training augmentation (off in the reported run)")
    ap.add_argument("--stats-scope", choices=["all", "train"], default="all",
                    help="compute min-max normalisation stats over all files (as in the reported run) "
                         "or over the training split only (stricter, avoids any test-set leakage)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--num-workers", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    # ---- Data -------------------------------------------------------------
    max_points = find_max_points(args.data_root)
    base = RoadPointCloudDataset(args.data_root)
    class_names = [base.id_to_label[i] for i in range(len(base.id_to_label))]
    if class_names != CLASS_NAMES:
        print(f"Note: detected classes {class_names} differ from the reported set {CLASS_NAMES}.")
    print(f"{len(base)} files, {len(class_names)} classes, padding to {max_points} points")

    n = len(base)
    n_train, n_val = int(SPLIT_RATIOS[0] * n), int(SPLIT_RATIOS[1] * n)
    n_test = n - n_train - n_val
    train_idx, val_idx, test_idx = (s.indices for s in random_split(range(n), [n_train, n_val, n_test]))

    fmin, fmax = compute_min_max(base, None if args.stats_scope == "all" else train_idx)
    ds = RoadPointCloudDataset(args.data_root, feature_min=fmin, feature_max=fmax)
    subset = lambda idx: torch.utils.data.Subset(ds, idx)
    train_ds = SubsetWithTransform(subset(train_idx), train_augmentation if args.augment else None)
    val_ds, test_ds = SubsetWithTransform(subset(val_idx)), SubsetWithTransform(subset(test_idx))
    print(f"Split: train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    collate = lambda b: collate_pad(b, max_points)
    mk = lambda d, shuffle: DataLoader(d, batch_size=args.batch_size, shuffle=shuffle,
                                       collate_fn=collate, num_workers=args.num_workers)
    train_loader, val_loader, test_loader = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

    # Inverse-frequency class weights from the training split
    counts = Counter(ds.labels[i] for i in train_idx)
    counts_t = torch.tensor([counts.get(i, 0) for i in range(len(class_names))], dtype=torch.float32)
    class_weights = counts_t.sum() / (len(class_names) * counts_t + 1e-10)
    for i, name in enumerate(class_names):
        print(f"  {name:<12} train samples {int(counts_t[i]):>4}  weight {class_weights[i]:.4f}")

    stats = {
        "feature_names": FEATURE_NAMES, "feature_min": fmin.tolist(), "feature_max": fmax.tolist(),
        "max_points": max_points, "class_names": class_names, "stats_scope": args.stats_scope,
    }
    with open(os.path.join(args.out_dir, "normalization_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ---- Model ------------------------------------------------------------
    model = SimplePointNet(len(FEATURE_NAMES), len(class_names), args.dropout).to(device)
    print(f"SimplePointNet with {count_parameters(model):,} trainable parameters")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=DEFAULT_HPARAMS["lr_scheduler_factor"],
                                  patience=DEFAULT_HPARAMS["lr_scheduler_patience"])

    # ---- Training loop ----------------------------------------------------
    weights_path = os.path.join(args.out_dir, "simplepointnet_best.pth")
    best_val_acc, best_val_loss, no_improve, history = -1.0, float("inf"), 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, correct, seen = 0.0, 0, 0
        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * points.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        train_loss, train_acc = run_loss / seen, correct / seen

        val_loss, y_true, y_pred = predict_loader(model, val_loader, device, criterion)
        val_acc = float((y_true == y_pred).mean())
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]})

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:>3}/{args.epochs}: train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        # Checkpoint on best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weights_path)

        # Early stopping on validation loss
        if val_loss < best_val_loss:
            best_val_loss, no_improve = val_loss, 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val-loss improvement for {args.patience} epochs)")
                break

    print(f"Training finished in {(time.time() - t0) / 60:.1f} min. Best val accuracy {best_val_acc:.4f}")

    # ---- Test evaluation with the best checkpoint --------------------------
    model.load_state_dict(torch.load(weights_path, map_location=device))
    test_loss, y_true, y_pred = predict_loader(model, test_loader, device, criterion)
    report = classification_report(y_true, y_pred, class_names)
    report.update({"test_loss": test_loss, "best_val_accuracy": best_val_acc,
                   "epochs_run": len(history), "hyperparameters": vars(args)})

    print(f"\nTest loss {test_loss:.4f} | accuracy {report['accuracy']:.4f} | "
          f"weighted P {report['weighted_precision']:.4f} R {report['weighted_recall']:.4f} "
          f"F1 {report['weighted_f1']:.4f} IoU {report['weighted_iou']:.4f}")
    print(format_confusion_matrix(report["confusion_matrix"], class_names))

    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved weights, stats, metrics and history to {args.out_dir}")


if __name__ == "__main__":
    main()
