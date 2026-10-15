# Methodology

This document describes the full pipeline behind the reported model, step by step. All settings match `src/config.py` and the original notebook in `notebooks/`.

## 1. Task

Given a point cloud of one road block captured by airborne laser scanning, predict which of six road configurations it shows:

| Class | Meaning |
|---|---|
| `2lanes` | Single carriageway with two lanes |
| `3lanes` | Single carriageway with three lanes |
| `crossing` | Junction or crossing area |
| `split4lanes` | Divided carriageway, four lanes in total |
| `split6lanes` | Divided carriageway, six lanes in total |
| `transition` | Section where the lane configuration changes |

## 2. Data

- **1,425 road blocks**, each stored as a `.npy` file with shape `(N, 22)`.
- `N` (points per block) varies from block to block, up to **16,661**.
- Labels come from the folder each file sits in.

Class distribution in the training split (1,140 blocks):

| Class | Training samples | Share | Class weight |
|---|---:|---:|---:|
| 2lanes | 373 | 32.7 % | 0.509 |
| 3lanes | 157 | 13.8 % | 1.210 |
| crossing | 105 | 9.2 % | 1.810 |
| split4lanes | 136 | 11.9 % | 1.397 |
| split6lanes | 171 | 15.0 % | 1.111 |
| transition | 198 | 17.4 % | 0.960 |

## 3. Preprocessing

### 3.1 Feature selection (22 → 11)

Only 11 of the 22 raw columns are used:

| Column index | Feature | Group |
|---:|---|---|
| 0, 1, 2 | local x, local y, local z | Geometry |
| 3, 4, 5 | red, green, blue | Colour |
| 9 | intensity | Laser return |
| 11 | planarity | Local shape |
| 12 | linearity | Local shape |
| 13 | sphericity | Local shape |
| 14 | verticality | Local shape |

The set was chosen experimentally: adding further features or removing any of these lowered validation accuracy.

### 3.2 Min-max normalisation

Each feature is scaled with `(value - min) / (max - min)` using one global min and max per feature. Without this, intensity (range 19,092 to 65,535) would dominate features such as planarity (range 0 to 1). This step gave the largest single improvement during development.

The values used by the shipped model are stored in `models/normalization_stats.json` so that new data can be scaled in exactly the same way.

### 3.3 Padding to a fixed size

Neural networks need equal-sized tensors in a batch, but every block has a different number of points. Each cloud is zero-padded to 16,661 points (the largest block in the dataset). Clouds larger than that would be truncated; none are in this dataset.

### 3.4 Train / validation / test split

A seeded random split (seed 42) with ratios 80 / 15 / 5:

| Split | Blocks | Purpose |
|---|---:|---|
| Train | 1,140 | Weight updates |
| Validation | 213 | Checkpoint selection, LR schedule, early stopping |
| Test | 72 | Final, one-off evaluation |

### 3.5 Augmentation (tested, then switched off)

Five augmentations were implemented and tested on the training split: random rotation about the vertical axis, Gaussian jitter, uniform scaling, random shift and random point dropout. All of them lowered validation accuracy, so the reported model was trained without augmentation. They remain available with `--augment`.

## 4. Model: SimplePointNet

A simplified version of PointNet (Qi et al., 2017), without the input and feature transform networks (T-Nets).

| Stage | Layer | Output shape | Parameters |
|---|---|---|---:|
| Input | | (B, 16661, 11) | |
| Shared MLP 1 | Conv1d(11→64, k=1) + BN + ReLU | (B, 64, N) | 896 |
| Shared MLP 2 | Conv1d(64→128, k=1) + BN + ReLU | (B, 128, N) | 8,576 |
| Shared MLP 3 | Conv1d(128→1024, k=1) + BN + ReLU | (B, 1024, N) | 134,144 |
| Global feature | Max pool over all points | (B, 1024) | 0 |
| FC 1 | Linear(1024→512) + BN + ReLU + Dropout(0.4) | (B, 512) | 525,824 |
| FC 2 | Linear(512→256) + BN + ReLU + Dropout(0.4) | (B, 256) | 131,840 |
| Output | Linear(256→6) | (B, 6) logits | 1,542 |
| **Total** | | | **802,822** |

**Why this works for point clouds:**

- A 1D convolution with kernel size 1 applies the *same* small network to every point independently, so the model learns what each point "looks like".
- Max pooling then keeps, for each of the 1,024 learned features, the strongest response anywhere in the block. The result does not depend on the order of the points, which is essential because a point cloud has no natural order.

## 5. Training

| Setting | Value |
|---|---|
| Loss | Cross-entropy with inverse-frequency class weights |
| Optimiser | Adam |
| Learning rate | 7 × 10⁻⁵ |
| Weight decay (L2) | 5 × 10⁻⁴ |
| Batch size | 8 |
| Dropout | 0.4 |
| LR scheduler | ReduceLROnPlateau on validation loss (factor 0.5, patience 5) |
| Max epochs | 200 |
| Early stopping | Stop after 30 epochs without validation-loss improvement |
| Checkpoint | Weights with the highest validation accuracy |
| Hardware | Single CUDA GPU |

The training code is written as a grid search over hyperparameters (`param_grid` in the notebook), so several combinations can be compared in one go; the reported run uses the final selected configuration above. Training stopped early at epoch 66.

## 6. Evaluation

The checkpoint with the best validation accuracy is evaluated once on the test split with:

- accuracy
- precision, recall and F1-score, weighted by class support
- IoU (Jaccard index), weighted by class support
- the confusion matrix

## 7. Notes on the setup

These points do not change the reported numbers, but anyone reproducing or extending the work should know them:

- **Normalisation statistics** were computed over all 1,425 files before splitting. Because the features have fixed physical ranges (for example 0 to 256 for colour, 0 to 1 for the shape features), the practical effect is negligible, but `train.py --stats-scope train` computes them from the training split only.
- **The split is random, not stratified.** With 72 test blocks, the smallest classes have only 7 to 8 test samples.
- **Exact reproduction of the split** requires running the notebook top to bottom, because earlier cells consume random numbers before the split. `src/train.py` uses the same method and seed but may draw a slightly different split.
