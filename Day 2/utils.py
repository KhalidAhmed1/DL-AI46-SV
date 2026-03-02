# ============================================================
# utils.py — Reproducibility helpers and shared utilities.
# We pin Python, NumPy, and PyTorch RNGs so every training run
# produces identical numbers given the same seed.
# ============================================================

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from config import SEED, RESULTS_DIR


def set_seed(seed: int = SEED) -> None:
    """
      - random.seed     → Python's built-in RNG used by data shuffling
      - np.random.seed  → NumPy (used in statistical operation)
      - torch.manual_seed → CPU tensor operations
      - cuda.manual_seed_all → every GPU visible to the process

      - deterministic / benchmark flags → make cuDNN choose the same
        algorithm every run (slight speed cost, huge reproducibility gain)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # turn off auto-tuner
    torch.backends.cudnn.benchmark = False   
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Reproducibility] Seed fixed to {seed} ✔")


def ensure_dirs() -> None:
    """Create output directories if they don't exist yet."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_loss_curves(
    train_losses: list,
    val_losses: list,
    title: str = "Loss Curves",
    filename: str = "loss.png",
) -> None:
    """
    Plot training vs. validation loss.
    which manage us to see if 
    learning is good  ->  train and val loss decrease together
    overfitting       ->  low train loss & high val loss
    underfitting      ->  both of them are high
    """
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss", color="royalblue")
    ax.plot(val_losses,   label="Val Loss",   color="tomato", linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {save_path}")


def plot_accuracy_curves(
    train_accs: list,
    val_accs: list,
    title: str = "Accuracy Curves",
    filename: str = "accuracy.png",
) -> None:
    """
    Plot training vs. validation accuracy.
    """
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_accs, label="Train Acc", color="seagreen")
    ax.plot(val_accs,   label="Val Acc",   color="darkorange", linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {save_path}")


def plot_comparison_bar(
    labels: list,
    val_accs: list,
    title: str = "Val Accuracy — Step Comparison",
    filename: str = "comparison.png",
) -> None:
    """
    Bar chart comparing validation accuracy for all of Golden Rule steps.
    """
    ensure_dirs()
    palette = sns.color_palette("viridis", len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, val_accs, color=palette, edgecolor="black", width=0.5)
    for bar, acc in zip(bars, val_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {save_path}")