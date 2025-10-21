#!/usr/bin/env python3
"""Entry point for running a federated learning client.

This script loads a configuration file (in YAML format) and starts
the Jetson federated client defined in ``federated/client.py``.  It
expects Flower to be installed.  The configuration should specify at
least the task name and training parameters.  Example usage::

    python run_fed_client.py --config configs/task_disaster.yaml --server localhost:8080

"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Any

import yaml

from federated.client import start_client


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a federated learning client.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML configuration file.")
    parser.add_argument("--server", type=str, default="localhost:8080", help="Address of the federated server.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    start_client(cfg, args.server)


if __name__ == "__main__":
    main()
