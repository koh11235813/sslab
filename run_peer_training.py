#!/usr/bin/env python3
"""Peer-to-peer federated training entry point.

This script launches a lightweight synchronous training loop that exchanges
model parameters directly between peers using the UDP-based ``AdHocTransport``.
Each device trains locally for a configurable number of epochs, applies optional
semantic compression to the model updates, broadcasts them to its peers, waits
for matching updates from every peer, averages the resulting weights, and
advances to the next round.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import yaml

# Ensure local imports work when running the script from the project root.
SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from semantic.encoder import dequantize_tensor, quantize_tensor, sparsify_topk
from task import load_task
from transport.ad_hoc import AdHocTransport


PeerAddress = Tuple[str, int]
TensorState = Dict[str, torch.Tensor]


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` and return the updated dict."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file supporting simple inheritance."""
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    inherits = cfg.pop("inherits", None)
    if inherits:
        base_cfg = load_config((path.parent / inherits).resolve())
        return deep_update(base_cfg, cfg)
    return cfg


def parse_peers(peers: str) -> List[PeerAddress]:
    """Parse a comma-separated list of peer addresses."""
    results: List[PeerAddress] = []
    for entry in peers.split(","):
        entry = entry.strip()
        if not entry:
            continue
        host, port_str = entry.split(":")
        resolved_host = socket.gethostbyname(host.strip())
        results.append((resolved_host, int(port_str)))
    return results


def ensure_training_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Populate ``cfg['training']`` with top-level fallbacks."""
    training_cfg = cfg.setdefault("training", {})
    for key in ("lr", "local_epochs", "batch_size", "rounds"):
        if key not in training_cfg and key in cfg:
            training_cfg[key] = cfg[key]
    training_cfg.setdefault("lr", 1e-3)
    training_cfg.setdefault("local_epochs", 1)
    training_cfg.setdefault("rounds", 3)
    return training_cfg


def get_semantic_settings(cfg: Dict[str, Any]) -> Tuple[str, float, int]:
    """Extract semantic compression parameters from the config."""
    semantic_cfg = cfg.get("semantic", {})
    mode = semantic_cfg.get("mode", "none")
    topk = semantic_cfg.get("topk", semantic_cfg.get("top_k", 0.1))
    bit_width = semantic_cfg.get("bit_width", semantic_cfg.get("bitwidth", 8))
    return mode, float(topk), int(bit_width)


def train_locally(
    model: torch.nn.Module,
    loaders: Dict[str, Any],
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    local_epochs: int,
    semantic_mode: str,
    topk: float,
) -> float:
    """Run local training and return the average training loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for _ in range(local_epochs):
        for batch in loaders["train"]:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.ndim > 2 and targets.ndim == 2:
                loss = loss_fn(outputs, targets.long())
            else:
                loss = loss_fn(outputs.squeeze(), targets.squeeze())
            loss.backward()
            if semantic_mode == "gradients_topk":
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data = sparsify_topk(param.grad.data, k=topk)
            optimizer.step()
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples if total_samples else 0.0


def capture_state_dict(model: torch.nn.Module) -> TensorState:
    """Return a CPU copy of the model state dictionary."""
    return {name: param.detach().cpu().clone() for name, param in model.state_dict().items()}


def encode_update(
    updated_state: TensorState,
    base_state: TensorState,
    semantic_mode: str,
    topk: float,
    bit_width: int,
) -> List[Dict[str, Any]]:
    """Prepare the model parameters for network transmission."""
    payload: List[Dict[str, Any]] = []
    for name, tensor in updated_state.items():
        record: Dict[str, Any] = {"name": name, "dtype": str(tensor.dtype)}
        if semantic_mode == "weights_quantize":
            q_tensor, scale, zero_point = quantize_tensor(tensor, bit_width=bit_width)
            record.update(
                {
                    "type": "quantized",
                    "array": q_tensor.numpy(),
                    "scale": float(scale),
                    "zero_point": float(zero_point),
                }
            )
        elif semantic_mode == "gradients_topk":
            delta = tensor - base_state[name]
            sparse_delta = sparsify_topk(delta, k=topk)
            record.update({"type": "sparse_delta", "array": sparse_delta.numpy()})
        else:
            record.update({"type": "full", "array": tensor.numpy()})
        payload.append(record)
    return payload


def decode_update(
    payload: Iterable[Dict[str, Any]],
    base_state: TensorState,
) -> TensorState:
    """Decode a payload received from a peer back into a parameter state."""
    state: TensorState = {}
    for record in payload:
        name = record["name"]
        dtype = base_state[name].dtype
        array = record["array"]
        record_type = record["type"]
        if record_type == "quantized":
            q_tensor = torch.from_numpy(array)
            tensor = dequantize_tensor(q_tensor, record["scale"], record["zero_point"]).to(dtype)
        elif record_type == "sparse_delta":
            delta = torch.from_numpy(array).to(dtype)
            tensor = base_state[name] + delta
        else:
            tensor = torch.from_numpy(array).to(dtype)
        state[name] = tensor
    return state


def aggregate_states(states: List[TensorState]) -> TensorState:
    """Average a list of parameter states."""
    if not states:
        raise ValueError("No states provided for aggregation.")
    averaged: TensorState = {
        name: torch.zeros_like(param) for name, param in states[0].items()
    }
    for state in states:
        for name, tensor in state.items():
            averaged[name] += tensor
    factor = 1.0 / float(len(states))
    for name in averaged:
        averaged[name] *= factor
    return averaged


def apply_state(model: torch.nn.Module, state: TensorState, device: torch.device) -> None:
    """Load an aggregated state dictionary into the model."""
    model_state = model.state_dict()
    for name, tensor in state.items():
        model_state[name] = tensor.to(model_state[name].device).type_as(model_state[name])
    model.load_state_dict(model_state)
    model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run peer-to-peer federated training.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--peers",
        required=True,
        help="Comma-separated list of peer addresses in the form host:port.",
    )
    parser.add_argument("--rounds", type=int, default=None, help="Override the number of communication rounds.")
    parser.add_argument("--port", type=int, default=None, help="Override the local UDP listening port.")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    training_cfg = ensure_training_defaults(cfg)
    semantic_mode, topk, bit_width = get_semantic_settings(cfg)

    if args.rounds is not None:
        training_cfg["rounds"] = args.rounds
    rounds = training_cfg.get("rounds", 3)

    peers = parse_peers(args.peers)
    if not peers:
        raise ValueError("At least one peer address must be provided.")

    p2p_cfg = cfg.get("p2p", {})
    local_port = args.port or int(p2p_cfg.get("local_port", 5000))
    listen_timeout = float(p2p_cfg.get("listen_timeout", 1.0))

    device_name = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    components = load_task(cfg)
    model = components["model"].to(device)
    loaders = components["loaders"]

    if cfg.get("task", {}).get("name") == "disaster":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.get("lr", 1e-3))
    local_epochs = int(training_cfg.get("local_epochs", 1))

    transport = AdHocTransport(local_port=local_port)
    transport.sock.settimeout(listen_timeout)

    try:
        base_state = capture_state_dict(model)
        expected_peers = set(peers)
        print(f"Starting P2P training for {rounds} rounds; listening on port {local_port}.")
        for round_idx in range(1, rounds + 1):
            train_loss = train_locally(
                model,
                loaders,
                loss_fn,
                optimizer,
                device,
                local_epochs,
                semantic_mode,
                topk,
            )
            updated_state = capture_state_dict(model)
            payload = encode_update(updated_state, base_state, semantic_mode, topk, bit_width)
            message = {
                "round": round_idx,
                "payload": payload,
            }
            data = pickle.dumps(message)
            for peer in expected_peers:
                transport.send(data, peer)

            received_states: List[TensorState] = []
            seen_peers: set[PeerAddress] = set()
            while len(seen_peers) < len(expected_peers):
                try:
                    packet, addr = transport.receive()
                except socket.timeout:
                    continue
                if addr not in expected_peers or addr in seen_peers:
                    continue
                incoming = pickle.loads(packet)
                if incoming.get("round") != round_idx:
                    continue
                peer_state = decode_update(incoming["payload"], base_state)
                received_states.append(peer_state)
                seen_peers.add(addr)

            aggregated_states = [updated_state] + received_states
            averaged_state = aggregate_states(aggregated_states)
            apply_state(model, averaged_state, device)
            base_state = capture_state_dict(model)
            print(
                f"Round {round_idx}/{rounds} complete: "
                f"train_loss={train_loss:.4f}, aggregated from {len(aggregated_states)} peers."
            )
    finally:
        transport.close()


if __name__ == "__main__":
    main()
