# ============================================================
# train.py — The Golden Rules of NN Training, all in one script.
#
# Run with:  python train.py
#
# This script follows the 4-step progression shown in the
# "Regularisation Techniques" diagram:
#
#   Step 1 — Sanity Check        (1 sample, simple model)
#   Step 2 — Establish Baseline  (full data, simple model)
#   Step 3 — Reduce Bias         (full data, complex model)
#   Step 4 — Reduce Variance     (full data, complex model + regularisation)
#
# Dataset: CIFAR-10 (60 000 colour images, 10 classes)
# ============================================================

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed
import matplotlib.pyplot as plt

# ── Local modules ──────────────────────────────────────────
from config import (
    SEED, DEVICE, RESULTS_DIR,
    STEP1_LR, STEP1_EPOCHS,
    STEP2_LR, STEP2_EPOCHS,
    STEP3_LR, STEP3_EPOCHS,
    STEP4_LR, STEP4_EPOCHS, STEP4_WEIGHT_DECAY,
)
from utils import set_seed, ensure_dirs, plot_loss_curves, plot_accuracy_curves, plot_comparison_bar
from dataset import get_loaders, get_single_sample_loader
from models import SimpleCNN, DeepCNN, count_parameters
from trainer import Trainer


# ──────────────────────────────────────────────────────────
# 0.  GLOBAL SETUP
# ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  CIFAR-10 — The Golden Rules of NN Training")
print("="*60)

# Pin ALL random sources so results are reproducible across runs.
set_seed(SEED)
ensure_dirs()
print(f"[Config] Device  : {DEVICE}")
print(f"[Config] Results : {os.path.abspath(RESULTS_DIR)}")


# ──────────────────────────────────────────────────────────
# Step 1 — Sanity Check  ("Few data, Simple model")
# ──────────────────────────────────────────────────────────
# Before training on the full dataset we must ensure 
# the forward pass works without any problem and the loss decrease
#
# if the model fail here -> there is a bug we must handle it
#
print("\n" + "━"*60)
print("STEP 1 — SANITY CHECK")
print("Goal: drive loss to ~0 on a SINGLE training sample.")
print("If this fails → stop and fix the bug before going further.")
print("━"*60)

# Fresh model, fresh seed → identical result every run
set_seed(SEED)
model_s1 = SimpleCNN()
print(f"[Model] SimpleCNN — {count_parameters(model_s1):,} trainable params")

optimizer_s1 = torch.optim.Adam(model_s1.parameters(), lr=STEP1_LR)
criterion     = nn.CrossEntropyLoss()

single_loader = get_single_sample_loader()
trainer_s1    = Trainer(model_s1, optimizer_s1, criterion)
history_s1    = trainer_s1.overfit_single_sample(single_loader, epochs=STEP1_EPOCHS)

# ── Plot sanity-check loss ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history_s1["train_loss"], color="royalblue")
ax.set_title("Step 1 — Sanity Check: Loss on Single Sample", fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.axhline(0.01, color="red", linestyle="--", label="Target < 0.01")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/step1_sanity_loss.png", dpi=150)
plt.close()
print("[Plot] step1_sanity_loss.png saved")


# ──────────────────────────────────────────────────────────
# 2.  STEP 2 — ESTABLISH BASELINE  ("Training data, Simple model")
# ──────────────────────────────────────────────────────────
# Now we train on the full dataset with the same simple model.
# Expected outcome: train loss ↓ but val accuracy will plateau
# We don't try to fix it here; we just measure the problem.
#
print("\n" + "━"*60)
print("STEP 2 — ESTABLISH BASELINE")
print("Goal: get a stable, reproducible number to beat in step 3.")
print("Expected: model underfits → val acc ~60–65%")
print("━"*60)

set_seed(SEED)
train_loader, val_loader = get_loaders(augment=False)

model_s2     = SimpleCNN()
optimizer_s2 = torch.optim.Adam(model_s2.parameters(), lr=STEP2_LR)
trainer_s2   = Trainer(model_s2, optimizer_s2, criterion)
history_s2   = trainer_s2.fit(train_loader, val_loader, epochs=STEP2_EPOCHS, save_best=False)

plot_loss_curves(history_s2["train_loss"], history_s2["val_loss"],
                 title="Step 2 — Baseline: Loss Curves", filename="step2_loss.png")
plot_accuracy_curves(history_s2["train_acc"], history_s2["val_acc"],
                     title="Step 2 — Baseline: Accuracy", filename="step2_acc.png")

baseline_val_acc = max(history_s2["val_acc"])
print(f"\n[Step 2 Result] Best Val Acc = {baseline_val_acc:.1f}%")


# ──────────────────────────────────────────────────────────
# 3.  STEP 3 — REDUCE BIAS  ("Training data, Complex model")
# ──────────────────────────────────────────────────────────
# We swap SimpleCNN for DeepCNN.
# More layers = more capacity = the model can fit the data.
# Expected outcome: both train AND val accuracy increase.
# BUT — train acc will climb much faster than val acc,
# the model starts to overfit (high variance).
#
print("\n" + "━"*60)
print("STEP 3 — REDUCE BIAS (Fix Underfitting)")
print("Goal: increase model capacity so val acc improves over baseline.")
print("Expected: train acc rises fast → overfitting signal appears.")
print("━"*60)

set_seed(SEED)
model_s3     = DeepCNN(use_dropout=False)   # no regularisation yet
optimizer_s3 = torch.optim.Adam(model_s3.parameters(), lr=STEP3_LR)
trainer_s3   = Trainer(model_s3, optimizer_s3, criterion)
history_s3   = trainer_s3.fit(train_loader, val_loader, epochs=STEP3_EPOCHS, save_best=False)
print(f"[Model] DeepCNN — {count_parameters(model_s3):,} trainable params")

plot_loss_curves(history_s3["train_loss"], history_s3["val_loss"],
                 title="Step 3 — Reduce Bias: Loss Curves", filename="step3_loss.png")
plot_accuracy_curves(history_s3["train_acc"], history_s3["val_acc"],
                     title="Step 3 — Reduce Bias: Accuracy", filename="step3_acc.png")

deep_val_acc = max(history_s3["val_acc"])
print(f"\n[Step 3 Result] Best Val Acc = {deep_val_acc:.1f}%")


# ──────────────────────────────────────────────────────────
# 4.  STEP 4 — REDUCE VARIANCE  ("Training data, Complex model, Regularisation")
# ──────────────────────────────────────────────────────────
# We keep the same DeepCNN but add three regularisation techniques:
#   1. Dropout (p=0.5)   — randomly zeros half the activations each step
#   2. Weight Decay (L2) — penalises large weights in the optimiser
#   3. Data Augmentation — random flips & crops = "free" extra data
#
# The combination should close the train–val gap and push val acc higher.
# This is the "Gold Standard" end goal.
#
print("\n" + "━"*60)
print("STEP 4 — REDUCE VARIANCE (Fix Overfitting)")
print("Regularisation toolkit:")
print("  • Dropout        (p=0.5)")
print("  • Weight Decay   (L2, λ=1e-4)")
print("  • Data Augmentation (random flip + crop + colour jitter)")
print("Goal: train–val gap narrows, val acc improves over Step 3.")
print("━"*60)

set_seed(SEED)
train_loader_aug, _ = get_loaders(augment=True)   

model_s4     = DeepCNN(use_dropout=True)
optimizer_s4 = torch.optim.Adam(
    model_s4.parameters(),
    lr=STEP4_LR,
    weight_decay=STEP4_WEIGHT_DECAY,   
)
trainer_s4   = Trainer(model_s4, optimizer_s4, criterion)
history_s4   = trainer_s4.fit(train_loader_aug, val_loader, epochs=STEP4_EPOCHS, save_best=True)

plot_loss_curves(history_s4["train_loss"], history_s4["val_loss"],
                 title="Step 4 — Reduce Variance: Loss Curves", filename="step4_loss.png")
plot_accuracy_curves(history_s4["train_acc"], history_s4["val_acc"],
                     title="Step 4 — Reduce Variance: Accuracy", filename="step4_acc.png")

reg_val_acc = max(history_s4["val_acc"])
print(f"\n[Step 4 Result] Best Val Acc = {reg_val_acc:.1f}%")


# ──────────────────────────────────────────────────────────
# 5.  FINAL SUMMARY — Compare all steps
# ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"  Step 2 — Baseline (SimpleCNN)               : {baseline_val_acc:.1f}%")
print(f"  Step 3 — Reduce Bias (DeepCNN, no reg)      : {deep_val_acc:.1f}%")
print(f"  Step 4 — Reduce Variance (DeepCNN + reg)    : {reg_val_acc:.1f}%")
improvement = reg_val_acc - baseline_val_acc
print(f"\n  Total improvement from Golden Rules         : +{improvement:.1f}%")
print("="*60)

# Bar chart comparing all steps
plot_comparison_bar(
    labels   = ["Baseline\n(Step 2)", "Reduce Bias\n(Step 3)", "Reduce Variance\n(Step 4)"],
    val_accs = [baseline_val_acc, deep_val_acc, reg_val_acc],
    title    = "CIFAR-10 Val Accuracy — Golden Rules Comparison",
    filename = "final_comparison.png",
)

print(f"\n[Done] All plots saved to ./{RESULTS_DIR}/")
print(f"[Done] Best model checkpoint → {RESULTS_DIR}/best_model.pth")