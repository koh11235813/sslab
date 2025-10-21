"""Message definitions for the semantic network transport layer.

Using plain Python dataclasses to define simple messages keeps the
implementation lightweight.  In a more sophisticated system you
might use Protobuf or another serialization mechanism for cross‑
language interoperability.  Here we define basic structures for
carrying model updates and semantic feature packets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union


@dataclass
class ModelDelta:
    """Represents a federated learning model update.

    Attributes
    ----------
    tensors : List[bytes]
        List of serialized tensors (e.g., pickled NumPy arrays or
        PyTorch tensors).
    task_name : str
        Name of the task that generated this update (e.g., ``"disaster"``).
    round : int
        Federated learning round number.
    bit_width : int
        Bit width used for quantization, if applicable.
    rtt_ms : float
        Observed round‑trip time in milliseconds.
    loss_rate : float
        Observed packet loss rate.
    """

    tensors: List[bytes]
    task_name: str
    round: int
    bit_width: int
    rtt_ms: float
    loss_rate: float


@dataclass
class FeaturePacket:
    """Carries a sequence of quantized feature codes.

    Attributes
    ----------
    codes : bytes
        Encoded representation of features or activations.
    task_name : str
        Name of the task these features correspond to.
    bit_width : int
        Bit width of the quantization used to produce ``codes``.
    """

    codes: bytes
    task_name: str
    bit_width: int
