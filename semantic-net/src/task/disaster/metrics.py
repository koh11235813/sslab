"""Evaluation metrics for the disaster analysis task.

This module implements simple segmentation metrics such as pixel
accuracy and Intersection over Union (IoU).  In practice you may
want to use more sophisticated metrics or integrate with existing
libraries such as ``torchmetrics``, but these functions provide a
lightweight starting point and do not require any additional
dependencies.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _flatten_predictions_and_targets(
    preds: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten predictions and targets to 1D tensors for metric computation.

    Parameters
    ----------
    preds : torch.Tensor
        Model output logits of shape (B, C, H, W).
    targets : torch.Tensor
        Ground truth labels of shape (B, H, W).

    Returns
    -------
    tuple
        Flattened predictions (class indices) and targets.
    """
    # Convert logits to predicted class indices.
    pred_labels = preds.argmax(dim=1).view(-1)
    true_labels = targets.view(-1)
    return pred_labels, true_labels


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute the pixel accuracy metric.

    Parameters
    ----------
    preds : torch.Tensor
        Model output logits of shape (B, C, H, W).
    targets : torch.Tensor
        Ground truth labels of shape (B, H, W).

    Returns
    -------
    float
        The ratio of correctly predicted pixels to the total number
        of pixels.
    """
    pred_labels, true_labels = _flatten_predictions_and_targets(preds, targets)
    correct = (pred_labels == true_labels).sum().item()
    total = true_labels.numel()
    return correct / total if total > 0 else 0.0


def intersection_over_union(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> float:
    """Compute the mean Intersection over Union (mIoU).

    Parameters
    ----------
    preds : torch.Tensor
        Model output logits of shape (B, C, H, W).
    targets : torch.Tensor
        Ground truth labels of shape (B, H, W).
    num_classes : int
        Number of classes.  Defaults to 2 for binary segmentation.

    Returns
    -------
    float
        The mean IoU across all classes.  Returns 0 if no class
        pixels are present.
    """
    pred_labels, true_labels = _flatten_predictions_and_targets(preds, targets)
    ious = []
    for cls in range(num_classes):
        pred_mask = pred_labels == cls
        target_mask = true_labels == cls
        intersection = torch.logical_and(pred_mask, target_mask).sum().item()
        union = torch.logical_or(pred_mask, target_mask).sum().item()
        if union == 0:
            # If there are no pixels of this class in target and prediction,
            # we don't include this class in the average.
            continue
        ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0
