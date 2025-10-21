"""Semantic encoding primitives for federated learning.

This module provides utilities to sparsify and quantize tensors
before sending them over the network.  Sparsification is performed
by selecting the top‑k elements by magnitude, while quantization
reduces the bit width of floating point values.  The functions here
operate on PyTorch tensors and return tensors suitable for
transmission.
"""

from __future__ import annotations

from typing import Tuple

import torch


def sparsify_topk(tensor: torch.Tensor, k: float = 0.1) -> torch.Tensor:
    """Keep only the top‑k fraction of elements by absolute value.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to sparsify.
    k : float
        Fraction of elements to keep (0 < k <= 1).  For example,
        ``k=0.1`` keeps the top 10% of elements by magnitude and sets
        all others to zero.

    Returns
    -------
    torch.Tensor
        A tensor of the same shape with most entries zeroed out.
    """
    if not 0 < k <= 1:
        raise ValueError("k must be between 0 and 1")
    tensor_flat = tensor.view(-1)
    n = tensor_flat.numel()
    # Determine the threshold for the top‑k elements.
    kth = max(1, int(n * k))
    if kth == n:
        return tensor.clone()
    values, indices = torch.topk(torch.abs(tensor_flat), kth)
    mask = torch.zeros_like(tensor_flat)
    mask[indices] = 1.0
    sparsified = tensor_flat * mask
    return sparsified.view_as(tensor)


def quantize_tensor(tensor: torch.Tensor, bit_width: int = 8) -> Tuple[torch.Tensor, float, float]:
    """Quantize a tensor to a lower bit precision using affine quantization.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to quantize.  Must be a floating point tensor.
    bit_width : int
        Number of bits to use for quantization (1–8).  Defaults to 8.

    Returns
    -------
    tuple
        A tuple ``(q_tensor, scale, zero_point)`` where ``q_tensor`` is
        the quantized tensor (dtype ``torch.int32``), and ``scale`` and
        ``zero_point`` are the quantization parameters needed for
        dequantization.
    """
    if bit_width < 1 or bit_width > 8:
        raise ValueError("bit_width must be between 1 and 8")
    qmin = 0
    qmax = 2 ** bit_width - 1
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    # Avoid division by zero if tensor is constant.
    if max_val == min_val:
        scale = 1.0
        zero_point = 0.0
        q_tensor = torch.zeros_like(tensor, dtype=torch.int32)
        return q_tensor, scale, zero_point
    scale = (max_val - min_val) / float(qmax - qmin)
    zero_point = qmin - min_val / scale
    q_tensor = (tensor / scale + zero_point).round().clamp(qmin, qmax).to(torch.int32)
    return q_tensor, scale, zero_point


def dequantize_tensor(q_tensor: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    """Dequantize a previously quantized tensor back to floating point.

    Parameters
    ----------
    q_tensor : torch.Tensor
        Quantized tensor of dtype ``torch.int32``.
    scale : float
        Quantization scale returned by ``quantize_tensor``.
    zero_point : float
        Quantization zero‑point returned by ``quantize_tensor``.

    Returns
    -------
    torch.Tensor
        The dequantized floating point tensor.
    """
    return scale * (q_tensor.to(torch.float32) - zero_point)
