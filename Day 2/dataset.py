# ============================================================
# dataset.py — Data loading & preprocessing for CIFAR-10.
#
# Dataset: CIFAR-10
#   - 60 000 images and 10 classes.
#   - Directly downloadable via torchvision 
# ============================================================

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from config import (BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, IMG_SIZE, SEED)


# ------------------------------------------------------------------
# Normalisation statistics for CIFAR-10 (pre-computed over train set)
# using the actual dataset mean/std instead of 0.5/0.5 
# ------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# Basic transform just resize and convert to tensor and normalise it
basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# We make here Augmented transform
# trying to make some random flips and crops
# this like we collect more data without collecting new samples
# we use it in traning only
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def get_loaders(
    augment: bool = False,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """
    Return: train_loader, val_loader

    Args:
        augment   : if True, training set uses aug_transform otherwise basic_transform
        data_root : Where to download the dataset.

    Returns:
        train_loader, val_loader 
    """
    train_tf = aug_transform if augment else basic_transform

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_tf
    )
    val_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=basic_transform
    )

    # We use a generator with our fixed seed
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        generator=g,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    print(f"[Dataset] Train samples : {len(train_set):,}")
    print(f"[Dataset] Val   samples : {len(val_set):,}")
    print(f"[Dataset] Augmentation  : {augment}")
    return train_loader, val_loader


def get_single_sample_loader(data_root: str = "./data") -> DataLoader:
    """
    Return a DataLoader with exactly 1 sample.

    This is for Step 1 (Sanity Check).
    - if the model can drive loss to ~0 on a single sample,
    the forward pass, loss, and backward pass are all correct.
    """
    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=basic_transform
    )
    single_sample = Subset(train_set, indices=[0])  # always the same sample
    loader = DataLoader(single_sample, batch_size=1, shuffle=False)
    label = train_set.classes[train_set.targets[0]]
    print(f"[Sanity Loader] 1 sample → class: '{label}' (index 0)")
    return loader


CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]