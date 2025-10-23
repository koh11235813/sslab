"""Model definitions for the network QoS prediction task.

The goal of this task is to predict future network performance
metrics (e.g. round‑trip time, throughput) based on recent history.
To keep dependencies minimal, we implement a small recurrent
neural network using PyTorch's ``nn.LSTM``.  This can be replaced
with more advanced architectures such as Temporal Convolutional
Networks (TCN) or Transformer models if desired.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class SimplePredictor(nn.Module):
    """A simple LSTM‑based predictor for time series forecasting.

    The model takes a sequence of past network measurements and
    predicts the next value in the sequence.  It consists of an LSTM
    followed by a linear layer.  The last hidden state of the LSTM
    is passed through the linear layer to produce the prediction.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1, output_size: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the predictor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, input_size), where B is the
            batch size and T is the sequence length.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, output_size), representing the
            predicted next value for each sequence in the batch.
        """
        output, (h_n, _) = self.lstm(x)
        # Use the hidden state from the final time step as the summary.
        last_hidden = h_n[-1]
        return self.fc(last_hidden)


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Factory function to construct the predictor model from a config.

    The configuration dictionary can specify ``input_size``,
    ``hidden_size``, ``num_layers`` and ``output_size`` under the
    ``task`` key.  Default values are provided if these keys are
    absent.

    Parameters
    ----------
    cfg : dict
        The full configuration dictionary.

    Returns
    -------
    nn.Module
        An instance of ``SimplePredictor`` configured according to ``cfg``.
    """
    task_cfg = cfg.get("task", {})
    input_size = task_cfg.get("input_size", 1)
    hidden_size = task_cfg.get("hidden_size", 32)
    num_layers = task_cfg.get("num_layers", 1)
    output_size = task_cfg.get("output_size", 1)
    return SimplePredictor(input_size, hidden_size, num_layers, output_size)
