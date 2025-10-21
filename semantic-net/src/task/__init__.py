"""Task package initialization and utilities.

This module defines a simple ``load_task`` function that constructs
the appropriate model, data loaders and metrics for a given task.  The
design encourages a modular structure where each task (e.g., disaster
analysis, network QoS prediction) provides its own ``model``,
``preprocess`` and ``metrics`` submodules.  By looking up the task
name in the configuration dictionary, we can dynamically import the
correct modules without hard‑coding their names here.  This makes it
easy to add new tasks: simply create a new folder under ``task/``
with ``model.py``, ``preprocess.py`` and ``metrics.py`` and adjust
``configs/*.yaml`` accordingly.

Example usage:

    from task import load_task
    cfg = {"task": {"name": "disaster"}}
    components = load_task(cfg)
    model = components["model"]
    loaders = components["loaders"]
    metrics = components["metrics"]

"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Dict


def load_task(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load task components based on the task name in ``cfg``.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.  Must contain the key
        ``"task" : {"name": ...}``.

    Returns
    -------
    dict
        A dictionary with keys ``model``, ``loaders`` and ``metrics``.
        ``model`` is a torch.nn.Module instance, ``loaders`` is a
        dictionary of PyTorch DataLoader objects (for training and
        validation), and ``metrics`` is a module providing metric
        functions/classes specific to the task.

    Raises
    ------
    ValueError
        If the task name is not recognized or the required modules
        cannot be imported.
    """
    task_name = cfg.get("task", {}).get("name")
    if not task_name:
        raise ValueError("Configuration must specify task.name")

    try:
        task_module: ModuleType = import_module(f"task.{task_name}")
    except Exception as exc:
        raise ValueError(f"Unknown or missing task module for '{task_name}': {exc}")

    # Import submodules lazily.  Each must define the expected API.
    model_module = import_module(f"task.{task_name}.model")
    preprocess_module = import_module(f"task.{task_name}.preprocess")
    metrics_module = import_module(f"task.{task_name}.metrics")

    # Build model and dataloaders using the provided config.
    model = model_module.build_model(cfg)
    loaders = preprocess_module.get_dataloaders(cfg)

    return {
        "model": model,
        "loaders": loaders,
        "metrics": metrics_module,
    }


__all__ = ["load_task"]