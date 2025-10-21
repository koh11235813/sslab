"""Semantic decoding utilities.

This module complements ``semantic.encoder`` by providing functions to
reconstruct tensors from their compressed or quantized representations.
Although sparsification itself does not need an explicit decoder (zero
entries remain zero), quantized tensors require the scale and
zero‑point parameters returned by the encoder.  These functions
mirror those in ``encoder`` for symmetry and ease of use.
"""

from __future__ import annotations

import torch

from .encoder import dequantize_tensor

__all__ = ["dequantize_tensor"]