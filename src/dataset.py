"""Data loading, normalisation, padding and (optional) augmentation for road point clouds."""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from config import SELECTED_FEATURE_INDICES


def set_seed(seed: int) -> None:
    """Make runs repeatable across Python, NumPy and PyTorch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_max_points(data_root: str) -> int:
    """Largest number of points in any .npy file under data_root (used as the padding size)."""
    max_points = 0
    for root, _, files in os.walk(data_root):
        for name in files:
            if name.endswith(".npy"):
                # mmap_mode avoids loading the whole array just to read its shape
                n = np.load(os.path.join(root, name), mmap_mode="r").shape[0]
                max_points = max(max_points, n)
    return max_points


class RoadPointCloudDataset(Dataset):
    """One sample = one road block point cloud stored as an (N, 22) .npy file.

    Expected layout (each sub-folder is one class):

        data_root/
            2lanes/       *.npy
            3lanes/       *.npy
            crossing/     *.npy
            split4lanes/  *.npy
            split6lanes/  *.npy
            transition/   *.npy

    Each item returns (points[N, 11] as float32, label as int64). Points keep their
    own length N; padding to a common size happens later in `collate_pad`.
    """

    def __init__(self, data_root, feature_min=None, feature_max=None, transform=None,
                 feature_indices=SELECTED_FEATURE_INDICES):
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"Dataset folder not found: {data_root}")

        self.feature_indices = list(feature_indices)
        self.feature_min = None if feature_min is None else np.asarray(feature_min, dtype=np.float32)
        self.feature_max = None if feature_max is None else np.asarray(feature_max, dtype=np.float32)
        self.transform = transform

        classes = sorted(d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)))
        self.label_to_id = {name: i for i, name in enumerate(classes)}
        self.id_to_label = {i: name for name, i in self.label_to_id.items()}

        self.file_paths, self.labels = [], []
        for name in classes:
            folder = os.path.join(data_root, name)
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".npy"):
                    self.file_paths.append(os.path.join(folder, fname))
                    self.labels.append(self.label_to_id[name])

    def __len__(self):
        return len(self.file_paths)

    def normalise(self, points: np.ndarray) -> np.ndarray:
        """Min-max scale each selected feature to roughly [0, 1]."""
        if self.feature_min is None or self.feature_max is None:
            return points
        value_range = self.feature_max - self.feature_min
        value_range = np.where(value_range == 0, 1e-6, value_range)
        return (points - self.feature_min) / value_range

    def __getitem__(self, idx):
        points = np.load(self.file_paths[idx]).astype(np.float32)
        points = points[:, self.feature_indices]
        points = self.normalise(points)
        if self.transform is not None:
            points = self.transform(points)
        return torch.from_numpy(np.ascontiguousarray(points)).float(), torch.tensor(self.labels[idx], dtype=torch.long)


def compute_min_max(dataset: RoadPointCloudDataset, indices=None):
    """Per-feature global min and max over the given samples (all samples if indices is None)."""
    indices = range(len(dataset)) if indices is None else indices
    fmin = np.full(len(dataset.feature_indices), np.inf, dtype=np.float32)
    fmax = np.full(len(dataset.feature_indices), -np.inf, dtype=np.float32)
    for i in indices:
        pts = np.load(dataset.file_paths[i]).astype(np.float32)[:, dataset.feature_indices]
        if pts.shape[0] == 0:
            continue
        fmin = np.minimum(fmin, pts.min(axis=0))
        fmax = np.maximum(fmax, pts.max(axis=0))
    return fmin, fmax


def collate_pad(batch, max_points: int):
    """Zero-pad (or truncate) every cloud in the batch to `max_points` so they stack into one tensor."""
    clouds, labels = zip(*batch)
    num_features = clouds[0].shape[1]
    padded = []
    for pc in clouds:
        n = pc.shape[0]
        if n > max_points:
            pc = pc[:max_points]
        elif n < max_points:
            pc = torch.cat([pc, torch.zeros(max_points - n, num_features, dtype=pc.dtype)], dim=0)
        padded.append(pc)
    return torch.stack(padded), torch.stack(labels)


class SubsetWithTransform(Dataset):
    """Wraps a Subset so augmentation can be applied to one split only."""

    def __init__(self, subset, transform=None):
        self.subset, self.transform = subset, transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        pc, label = self.subset[idx]
        if self.transform is not None:
            pc = torch.from_numpy(self.transform(pc.numpy())).float()
        return pc, label


# ---------------------------------------------------------------------------
# Augmentation (tested in the experiments, but it lowered accuracy on this
# dataset, so it is switched OFF by default; enable with --augment in train.py)
# ---------------------------------------------------------------------------

def rotate_z(pc):
    a = np.random.uniform() * 2 * np.pi
    c, s = np.cos(a), np.sin(a)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pc[:, :3] = pc[:, :3] @ rot
    return pc


def jitter(pc, sigma=0.0005, clip=0.001):
    pc[:, :3] += np.clip(sigma * np.random.randn(pc.shape[0], 3), -clip, clip)
    return pc


def scale(pc, low=0.99, high=1.01):
    pc[:, :3] *= np.random.uniform(low, high)
    return pc


def shift(pc, shift_range=0.005):
    pc[:, :3] += np.random.uniform(-shift_range, shift_range, size=(1, 3))
    return pc


def point_dropout(pc, max_ratio=0.01):
    n_drop = int(pc.shape[0] * np.random.uniform(0, max_ratio))
    if n_drop > 0:
        pc = np.delete(pc, np.random.choice(pc.shape[0], n_drop, replace=False), axis=0)
    return pc


def train_augmentation(pc):
    for fn in (rotate_z, jitter, scale, shift, point_dropout):
        pc = fn(pc)
    return pc
