#!/usr/bin/env python3
"""Entry point for running a federated learning server.

This script starts a Flower federated server using the simple FedAvg
strategy defined in ``federated/server.py``.  It requires Flower to
be installed.  Example usage::

    python run_fed_server.py --port 8080 --rounds 5

"""

from __future__ import annotations

import argparse

from federated.server import start_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a federated learning server.")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated training rounds.")
    args = parser.parse_args()

    cfg = {"federated": {"min_clients": 1}}
    server_address = f"0.0.0.0:{args.port}"
    start_server(cfg, server_address, num_rounds=args.rounds)


if __name__ == "__main__":
    main()
