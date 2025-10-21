"""Data preprocessing utilities for the disaster analysis task.

This module defines a minimal dataset and loader setup for a
segmentation problem.  For demonstration purposes, we provide a
``DummySegDataset`` that generates random RGB images and binary
segmentation masks.  In a real application you should replace
``DummySegDataset`` with a dataset class that reads actual images
and annotations from disk.

The ``get_dataloaders`` function returns PyTorch DataLoader objects
for training and validation.  It accepts a configuration dictionary
and looks up task‑specific settings such as batch size and image
size.  If these keys are absent, sensible defaults are used.
"""

from __future__ import annotations

import random
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class DummySegDataset(Dataset):
    """A synthetic dataset for binary segmentation.

    Each sample consists of a randomly generated RGB image and a
    corresponding segmentation mask with two classes (foreground and
    background).  The random data makes this dataset unsuitable for
    real training, but it serves to illustrate the data loading
    pipeline without needing external files.
    """

    def __init__(self, num_samples: int, image_size: int, num_classes: int = 2) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create a random RGB image and a random segmentation mask.
        img = torch.rand(3, self.image_size, self.image_size)
        mask = torch.randint(0, self.num_classes, (self.image_size, self.image_size))
        return img, mask


def get_dataloaders(cfg: Dict[str, object]) -> Dict[str, DataLoader]:
    """Construct training and validation dataloaders from a config.

    The configuration dictionary may include the following keys under
    ``"training"`` and ``"task"``:

    ``batch_size``: Size of the mini‑batches.  Defaults to 8.
    ``image_size``: Height/width of the input images.  Defaults to 128.
    ``num_samples``: Total number of samples per split.  Defaults to 100.
    ``val_split``: Fraction of samples reserved for validation.  Defaults to 0.2.

    Parameters
    ----------
    cfg : dict
        The full configuration dictionary.

    Returns
    -------
    dict
        A dictionary with keys ``train`` and ``val``, each mapping to a
        PyTorch DataLoader.
    """
    training_cfg = cfg.get("training", {})
    batch_size = training_cfg.get("batch_size", 8)
    image_size = training_cfg.get("image_size", 128)
    num_samples = training_cfg.get("num_samples", 100)
    val_split = training_cfg.get("val_split", 0.2)
    num_classes = cfg.get("task", {}).get("num_classes", 2)

    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val

    train_dataset = DummySegDataset(num_train, image_size, num_classes)
    val_dataset = DummySegDataset(num_val, image_size, num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader}
