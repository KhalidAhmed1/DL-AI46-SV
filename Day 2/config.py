# ============================================================
# config.py — Central configuration for ALL hyperparameters.
# ============================================================

import torch
SEED = 42


# Device use GPU if available, otherwise fall back to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset: we use CIFAR-10 -> 60 000 images and 10 classes.
DATASET        = "CIFAR10"
NUM_CLASSES    = 10
IMG_SIZE       = 32         
IN_CHANNELS    = 3           


HIDDEN_DIM     = 256
# used in Step of regularization
DROPOUT_RATE   = 0.5         

# Training loop shared settings
BATCH_SIZE     = 128
NUM_WORKERS    = 0
PIN_MEMORY     = True


# ============ Learning rates & epochs for each step =============

# - Steps of "Golden Standard"
# Step 1 — Sanity Check: 1 sample (Code Verification)
# we make alot of epochs here to make overfit
STEP1_LR       = 1e-3
STEP1_EPOCHS   = 200

# Step 2 — Establish Baseline: full training set, simple model
STEP2_LR       = 1e-3
STEP2_EPOCHS   = 10

# Step 3 — Reduce Bias: deeper model, same data
STEP3_LR       = 1e-3
STEP3_EPOCHS   = 20

# Step 4 — Reduce Variance: add regularisation (dropout + weight decay)
STEP4_LR       = 1e-3
STEP4_WEIGHT_DECAY = 1e-4
STEP4_EPOCHS   = 30

# Output paths — all artefacts land in one place.
RESULTS_DIR    = "results"
MODEL_PATH     = f"{RESULTS_DIR}/best_model.pth"