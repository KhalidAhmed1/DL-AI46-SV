# ============================================================
# models.py — 2 CNN architectures following the Golden Rules.
#
# SimpleCNN  → that's met Step 2 "Establish Baseline" 
# DeepCNN    → that's met Step 3 "Reduce Bias" + Step 4 "Reduce Variance" 
# ============================================================

import torch
import torch.nn as nn
from config import NUM_CLASSES, IN_CHANNELS, DROPOUT_RATE


class SimpleCNN(nn.Module):
    """
    the baseline model: 2 conv blocks + 1 fully connected head.
    """

    def __init__(self) -> None:
        super().__init__()

        # extract low level features like edges and colours
        self.block1 = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),       
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), 
        )

        # combine these low level features into shapes
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),       
        )

        # classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """here we go to the forward path"""
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


class DeepCNN(nn.Module):
    """
    This is Improved model used in step 3 & 4
    - Step 3 (Fix Underfitting)  : use_dropout=False
    - Step 4 (Fix Overfitting)   : use_dropout=True

    4 conv blocks + Dropout option.
    using AdaptiveAvgPool2d here: to make model less sensitive to the position
    only care with feature is existed or not (not the position)

    Architecture:
        Block1: Conv(3→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool
        Block2: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool
        Block3: Conv(128→256) → BN → ReLU → MaxPool
        Block4: Conv(256→256) → BN → ReLU → AvgPool (global)
        Head  : FC(256 → 512) → ReLU → [Dropout] → FC(512 → 10)
    """

    def __init__(self, use_dropout: bool = False) -> None:
        super().__init__()
        self.use_dropout = use_dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),          nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),   # 32→16
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),   # 16→8
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),   # 8→4
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),           # 4→1 (global average pooling)
        )

        # Dropout is regularization technique: 
        # randomly zeros activations during training, forcing the network to not depend neurons.
        p = DROPOUT_RATE if use_dropout else 0.0
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """count trainable parameters before training"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)