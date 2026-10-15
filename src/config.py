"""Shared settings for the road point cloud classification project.

All values here mirror the settings used in the original experiment notebook
(notebooks/road_pointcloud_classification.ipynb).
"""

# Columns kept from each raw .npy point (the raw files hold 22 features per point).
# Order matters: the trained model expects exactly these 11 columns in this order.
SELECTED_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 9, 11, 12, 13, 14]

FEATURE_NAMES = [
    "local_x",
    "local_y",
    "local_z",
    "red",
    "green",
    "blue",
    "intensity",
    "planarity",
    "linearity",
    "sphericity",
    "verticality",
]

# Class names come from the sub-folder names, sorted alphabetically.
CLASS_NAMES = ["2lanes", "3lanes", "crossing", "split4lanes", "split6lanes", "transition"]

# Hyperparameters of the best (and reported) run.
DEFAULT_HPARAMS = {
    "learning_rate": 7e-5,
    "weight_decay": 5e-4,
    "batch_size": 8,
    "dropout_rate": 0.4,
    "num_epochs": 200,
    "early_stop_patience": 30,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 5,
}

# Train / validation / test split ratios.
SPLIT_RATIOS = (0.80, 0.15, 0.05)

SEED = 42
