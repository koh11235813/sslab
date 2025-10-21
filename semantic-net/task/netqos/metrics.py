"""Evaluation metrics for the network QoS prediction task.

This module provides simple regression metrics including mean
absolute error (MAE) and root mean squared error (RMSE).  These
functions operate on tensors and return scalar floats.  In a
production environment you might want to use ``torchmetrics`` or
scikit‑learn for more comprehensive metrics, but these implementations
avoid external dependencies.
"""

from __future__ import annotations

import torch


def mean_absolute_error(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute the mean absolute error between predictions and targets.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted values of shape (B, D) or (B,) where B is the batch size.
    targets : torch.Tensor
        True values with the same shape as ``preds``.

    Returns
    -------
    float
        The mean absolute error over the batch and output dimensions.
    """
    return torch.mean(torch.abs(preds - targets)).item()


def root_mean_squared_error(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute the root mean squared error between predictions and targets.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted values of shape (B, D) or (B,).
    targets : torch.Tensor
        True values with the same shape as ``preds``.

    Returns
    -------
    float
        The root mean squared error over the batch and output dimensions.
    """
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()
