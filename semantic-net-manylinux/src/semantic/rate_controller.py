"""Simple rate controller for semantic communication.

Federated learning in bandwidth‑limited or unreliable networks
benefits from adapting the amount of data transmitted based on
observed network conditions.  This module provides a toy example of
such a controller.  It exposes a ``RateController`` class that
maintains the current bitrate (in terms of sparsification level or
quantization bit width) and adjusts it in response to round‑trip
times (RTT) and packet loss rates.

The adaptation strategy here is intentionally simple: if RTT or
loss exceed configurable thresholds, the controller reduces the
bitrate; otherwise it gradually increases it up to a maximum.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RateController:
    """A basic controller for selecting communication parameters.

    Attributes
    ----------
    min_bit_width : int
        The minimum quantization bit width allowed (e.g., 2).
    max_bit_width : int
        The maximum quantization bit width allowed (e.g., 8).
    current_bit_width : int
        The current quantization bit width in use.
    rtt_threshold_ms : float
        RTT threshold above which the controller reduces bit width.
    loss_threshold : float
        Packet loss threshold above which the controller reduces bit width.
    """

    min_bit_width: int = 2
    max_bit_width: int = 8
    current_bit_width: int = 8
    rtt_threshold_ms: float = 200.0
    loss_threshold: float = 0.1

    def update(self, rtt_ms: float, loss_rate: float) -> int:
        """Update the bit width based on observed network conditions.

        Parameters
        ----------
        rtt_ms : float
            Observed round‑trip time in milliseconds.
        loss_rate : float
            Observed packet loss rate (0–1).

        Returns
        -------
        int
            The updated bit width to use for quantization.
        """
        # If RTT or loss is high, decrease bit width to reduce traffic.
        if rtt_ms > self.rtt_threshold_ms or loss_rate > self.loss_threshold:
            self.current_bit_width = max(self.min_bit_width, self.current_bit_width - 1)
        else:
            # Otherwise, cautiously increase bit width up to the maximum.
            if self.current_bit_width < self.max_bit_width:
                self.current_bit_width += 1
        return self.current_bit_width
