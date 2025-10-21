"""Model definitions for the disaster task.

This module provides a very lightweight segmentation network suitable
for running on NVIDIA Jetson devices.  It is intentionally simple to
serve as a template; you can replace it with a more sophisticated
architecture such as MobileNet, UNet or SegFormer if needed.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleSegNet(nn.Module):
    """A minimal convolutional neural network for binary segmentation.

    The network downsamples the input, processes it through a stack of
    convolutions, and then upsamples back to the original resolution
    using bilinear interpolation.  This is not intended to provide
    state‑of‑the‑art performance but to illustrate the mechanics of a
    segmentation model in a resource‑constrained environment.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the segmentation network.

        Args:
            x: input tensor of shape (batch, channels, height, width).

        Returns:
            Logits tensor of shape (batch, num_classes, height, width).
        """
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits


def build_model(cfg: dict) -> nn.Module:
    """Factory function to construct the segmentation model from a config.

    The configuration dictionary is expected to contain `num_classes` and
    possibly other task‑specific parameters in the future.
    """
    num_classes = cfg.get("task", {}).get("num_classes", 2)
    return SimpleSegNet(num_classes=num_classes)