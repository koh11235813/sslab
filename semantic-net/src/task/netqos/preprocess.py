"""Data preprocessing utilities for the network QoS prediction task.

This module defines a synthetic time series dataset and functions to
construct PyTorch DataLoaders for training and validation.  In
practice you would replace ``DummyTimeSeriesDataset`` with a
dataset that reads measurements from real network logs.  The dummy
dataset generates sequences of random values and sets the target to
be the value at the next time step.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DummyTimeSeriesDataset(Dataset):
    """Synthetic dataset for one‑step time series forecasting.

    Each item in the dataset consists of a sequence of ``seq_length``
    random numbers and a target equal to the next number in the
    sequence.  This simple pattern allows us to demonstrate how to
    prepare data for an LSTM without relying on external files.
    """

    def __init__(self, num_samples: int, seq_length: int, input_size: int = 1) -> None:
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.input_size = input_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate a random sequence using a simple autoregressive process.
        # Start with a random seed and then add small Gaussian noise.
        seq = np.zeros((self.seq_length + 1, self.input_size), dtype=np.float32)
        seq[0] = np.random.randn(self.input_size).astype(np.float32)
        for t in range(1, self.seq_length + 1):
            seq[t] = seq[t - 1] + 0.1 * np.random.randn(self.input_size)
        # The input is the first ``seq_length`` values and the target is the next value.
        x = torch.tensor(seq[: self.seq_length], dtype=torch.float32)
        y = torch.tensor(seq[self.seq_length], dtype=torch.float32)
        return x, y


def get_dataloaders(cfg: Dict[str, object]) -> Dict[str, DataLoader]:
    """Construct training and validation dataloaders for the time series task.

    Configuration keys under ``training``:
      - ``batch_size``: Size of the mini‑batches (default: 16).
      - ``seq_length``: Length of input sequences (default: 10).
      - ``num_samples``: Total number of sequences (default: 1000).
      - ``val_split``: Fraction reserved for validation (default: 0.2).

    Parameters
    ----------
    cfg : dict
        The full configuration dictionary.

    Returns
    -------
    dict
        A dictionary with ``train`` and ``val`` DataLoaders.
    """
    training_cfg = cfg.get("training", {})
    batch_size = training_cfg.get("batch_size", 16)
    seq_length = training_cfg.get("seq_length", 10)
    num_samples = training_cfg.get("num_samples", 1000)
    val_split = training_cfg.get("val_split", 0.2)
    input_size = cfg.get("task", {}).get("input_size", 1)

    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val

    train_dataset = DummyTimeSeriesDataset(num_train, seq_length, input_size)
    val_dataset = DummyTimeSeriesDataset(num_val, seq_length, input_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader}
