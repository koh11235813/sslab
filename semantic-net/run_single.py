#!/usr/bin/env python3
"""Run standalone training on a single Jetson device.

This script loads a task specified via command line arguments, builds
the corresponding model and data loaders, and performs a simple
training loop.  The goal is to enable quick experimentation on a
single device before scaling up to federated learning.  It supports
both the disaster segmentation and network QoS prediction tasks
defined in this repository.

Example:

    python run_single.py --task disaster --epochs 5
    python run_single.py --task netqos --epochs 10 --batch_size 32

"""

from __future__ import annotations

import argparse
from typing import Dict, Any

import torch

from task import load_task
from semantic.encoder import sparsify_topk, quantize_tensor


def train_one_epoch(model: torch.nn.Module, loaders: Dict[str, Any], loss_fn, optimizer, device: torch.device, cfg: Dict[str, Any]) -> float:
    """Train the model for a single epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    count = 0
    semantic_mode = cfg.get("semantic", {}).get("mode", "none")
    topk = cfg.get("semantic", {}).get("topk", 0.1)
    for inputs, targets in loaders["train"]:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if cfg.get("task", {}).get("name") == "disaster":
            loss = loss_fn(outputs, targets.long())
        else:
            loss = loss_fn(outputs.squeeze(), targets.squeeze())
        loss.backward()
        if semantic_mode == "gradients_topk":
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data = sparsify_topk(p.grad.data, k=topk)
        optimizer.step()
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        count += batch_size
    return total_loss / count if count > 0 else 0.0


def validate(model: torch.nn.Module, loaders: Dict[str, Any], loss_fn, device: torch.device, cfg: Dict[str, Any]) -> float:
    """Evaluate the model on the validation set and return the average loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in loaders["val"]:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            if cfg.get("task", {}).get("name") == "disaster":
                loss = loss_fn(outputs, targets.long())
            else:
                loss = loss_fn(outputs.squeeze(), targets.squeeze())
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            count += batch_size
    return total_loss / count if count > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Single device training for semantic network tasks.")
    parser.add_argument("--task", type=str, required=True, choices=["disaster", "netqos"], help="Name of the task to run.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training (overrides config).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device.")
    args = parser.parse_args()

    # Build a configuration dictionary.  In a real application you might
    # load YAML files, but here we construct the relevant fields
    # directly for simplicity.
    cfg: Dict[str, Any] = {
        "task": {"name": args.task},
        "training": {
            "local_epochs": args.epochs,
        },
        "semantic": {"mode": "none"},
    }
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr

    device = torch.device(args.device)
    # Load task components after constructing cfg to ensure overridden batch_size and lr are respected.
    components = load_task(cfg)
    model = components["model"].to(device)
    loaders = components["loaders"]

    # Choose loss function based on task
    loss_fn = torch.nn.CrossEntropyLoss() if args.task == "disaster" else torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr or 1e-3)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, loaders, loss_fn, optimizer, device, cfg)
        val_loss = validate(model, loaders, loss_fn, device, cfg)
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


if __name__ == "__main__":
    main()
