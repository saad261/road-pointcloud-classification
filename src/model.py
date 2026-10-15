"""SimplePointNet: a compact PointNet-style classifier for whole point clouds."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointNet(nn.Module):
    """Simplified PointNet (Qi et al., 2017) without the T-Net alignment blocks.

    Three stages:
      1. Per-point feature learning with shared MLPs (Conv1d, kernel size 1).
      2. Global aggregation with max pooling over all points (order-invariant).
      3. A fully connected classification head with BatchNorm and Dropout.

    Input:  (batch, num_points, num_features)
    Output: (batch, num_classes) raw logits
    """

    def __init__(self, num_features: int = 11, num_classes: int = 6, dropout_rate: float = 0.4):
        super().__init__()

        # Shared per-point MLP: 11 -> 64 -> 128 -> 1024
        self.mlp1_conv = nn.Conv1d(num_features, 64, kernel_size=1)
        self.mlp1_bn = nn.BatchNorm1d(64)
        self.mlp2_conv = nn.Conv1d(64, 128, kernel_size=1)
        self.mlp2_bn = nn.BatchNorm1d(128)
        self.mlp3_conv = nn.Conv1d(128, 1024, kernel_size=1)
        self.mlp3_bn = nn.BatchNorm1d(1024)

        # Classification head: 1024 -> 512 -> 256 -> num_classes
        self.fc1 = nn.Linear(1024, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)                       # (B, N, F) -> (B, F, N)
        x = F.relu(self.mlp1_bn(self.mlp1_conv(x)))  # (B, 64, N)
        x = F.relu(self.mlp2_bn(self.mlp2_conv(x)))  # (B, 128, N)
        x = F.relu(self.mlp3_bn(self.mlp3_conv(x)))  # (B, 1024, N)
        x = torch.max(x, dim=2)[0]                   # (B, 1024) global feature
        x = self.dropout1(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.dropout2(F.relu(self.fc2_bn(self.fc2(x))))
        return self.fc3(x)                           # logits


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
