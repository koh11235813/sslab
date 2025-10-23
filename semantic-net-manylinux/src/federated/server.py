"""Federated learning server.

This module defines a minimal Flower server that aggregates model
updates from clients using the standard FedAvg algorithm.  It is
intended for experimentation with the semantic network and can be
extended to include alternative aggregation strategies or secure
aggregation mechanisms.
"""

from __future__ import annotations

from typing import Dict, Optional

try:
    import flwr as fl  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The federated server requires Flower (flwr) to be installed. "
        "Install it via `pip install flwr`."
    ) from exc


def start_server(cfg: Dict[str, object], server_address: str, num_rounds: int = 3) -> None:
    """Start the federated learning server.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary, currently unused but provided for symmetry
        with the client.
    server_address : str
        Address on which the server listens, e.g. ``"localhost:8080"``.
    num_rounds : int
        Number of federated training rounds to perform.  Defaults to 3.
    """
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=cfg.get("federated", {}).get("min_clients", 1)
    )
    fl.server.start_server(server_address, config={"num_rounds": num_rounds}, strategy=strategy)
