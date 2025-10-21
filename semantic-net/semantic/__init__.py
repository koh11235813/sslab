"""Semantic encoding and decoding utilities.

This package contains functions for compressing and quantizing
model parameters or intermediate features prior to transmission over
bandwidth‑limited networks.  It also provides simple rate
controllers to adapt the bit‑rate based on network conditions.
"""

__all__ = ["encoder", "decoder", "rate_controller"]