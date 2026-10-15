# Dataset

The dataset is **not included** in this repository. It was provided for the course
*Machine Learning for Civil Engineering* at RWTH Aachen University and is not
redistributed here.

## Expected layout

Place the data so the folder structure looks like this (folder names become class labels):

```
data/dataset/
├── 2lanes/        *.npy
├── 3lanes/        *.npy
├── crossing/      *.npy
├── split4lanes/   *.npy
├── split6lanes/   *.npy
└── transition/    *.npy
```

## File format

- One `.npy` file = one road block, stored as a NumPy array of shape `(N, 22)`:
  `N` points (varies per file), 22 features per point.
- The model uses 11 of the 22 columns (indices `0, 1, 2, 3, 4, 5, 9, 11, 12, 13, 14`):
  local x, local y, local z, red, green, blue, intensity, planarity, linearity,
  sphericity, verticality.
- In the version used here: 1,425 files, with up to 16,661 points per file.

To use your own data, keep the same column order, or edit
`SELECTED_FEATURE_INDICES` in `src/config.py`.
