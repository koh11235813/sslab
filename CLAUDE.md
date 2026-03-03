# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Interaction contract
- If requirements are ambiguous or underspecified, stop and ask 1–3 targeted questions before proceeding.
- Before making any irreversible change (deletes, migrations, dependency upgrades, infra changes), ask for explicit confirmation.
- Never assume environment details (OS, shell, package manager, project conventions). Ask or infer only from repo evidence.
- Start each task by restating: Goal, Non-goals, Constraints, Success criteria (brief).
- When multiple approaches exist, present 2 options with tradeoffs, then ask which to take.

## Project Overview

Federated semantic communication experiments for bandwidth-constrained edge devices (primarily NVIDIA Jetson). The repo combines task-specific PyTorch models with semantic compression (sparsification/quantization) and Flower-based federated learning.

## Repository Structure

The repo contains three independent subprojects, each with its own `pyproject.toml`, `.venv/`, and `uv.lock`:

- **`semantic-net/`** — Jetson-targeted (aarch64) variant. Python >=3.9. Has Dockerfile for JetPack (L4T r36.4.0).
- **`semantic-net-manylinux/`** — x86_64 variant supporting CPU, CUDA 12.4, and ROCm 6.1. Python 3.10–3.11. Includes additional SegFormer training scripts and dataset utilities.
- **`federated-learning/`** — Standalone Flower runner with dependency metadata. Python 3.10. Adds TensorFlow alongside PyTorch. Currently a scaffold (`main.py` prints a greeting).

`semantic-net/` and `semantic-net-manylinux/` share an identical core architecture under `src/`:
- `src/run_single.py`, `src/run_fed_client.py`, `src/run_fed_server.py` — entry points
- `src/semantic/` — encoder (sparsify_topk, quantize_tensor), decoder (dequantize_tensor), rate_controller
- `src/federated/` — Flower client (`JetsonClient`) and FedAvg server wrapper
- `src/task/` — pluggable tasks discovered by `task.load_task(cfg)` via dynamic import. Each task has `model.py` (`build_model`), `preprocess.py` (`get_dataloaders`), `metrics.py`
- `src/transport/` — experimental UDP transport (`AdHocTransport`) and `ModelDelta`/`FeaturePacket` dataclasses
- `configs/` — YAML presets with three top-level keys: `task`, `training`, `semantic`

`semantic-net-manylinux/` additionally has:
- `src/train_segformer_b{0,1,2,3}.py` — SegFormer fine-tuning on RescueNet patches
- `src/eval_segformer.py`, `src/eval_segformer_fp16.py` — evaluation scripts
- `src/dataset/` — RescueNet patch loader and resizing helpers
- `src/calculate_model_cost.py`, `src/precompute_values.py`, `src/train_model_selector.py` — model selection utilities
- `scripts/smoke.py` — quick PyTorch install verification

## Build & Run Commands

Each subproject is managed independently with `uv`. Always `cd` into the subproject first.

### semantic-net (Jetson)
```bash
cd semantic-net
uv sync --frozen                       # base deps
uv sync --frozen --extra jetson        # with Jetson PyTorch wheels
uv run --frozen python src/run_single.py --task disaster --epochs 1   # smoke test
```

### semantic-net-manylinux (x86_64)
```bash
cd semantic-net-manylinux
uv sync --frozen --extra cpu           # or --extra cu124 / --extra rocm
uv run --frozen python scripts/smoke.py                               # verify torch
uv run --frozen python src/run_single.py --task disaster --epochs 1   # smoke test
uv run --frozen python src/train_segformer_b3.py --data_root dataset/RescueNet_patches --epoch 50 --batch_size 8
```

### federated-learning
```bash
cd federated-learning
uv sync                                # or --extra jetson / --extra cpu / --extra cu124 / --extra rocm
uv run python main.py
```

### Federated simulation (in either semantic-net variant)
```bash
# Terminal 1
uv run --frozen python src/run_fed_server.py --port 8080 --rounds 3
# Terminal 2
uv run --frozen python src/run_fed_client.py --config configs/task_disaster.yaml --server localhost:8080
```

### Docker (semantic-net for Jetson)
```bash
docker build -t semantic-net:jp36 .
docker run --rm -it --runtime nvidia --network host semantic-net:jp36 bash
```

### Tests
No automated test suite exists yet. Use `uv run pytest` once tests are added under `tests/`.

## Key Conventions

- **Coding style**: PEP 8, 4-space indent, snake_case, type annotations (follow `federated/client.py` pattern).
- **Config inheritance**: Start from `configs/base.yaml`, override only task-specific sections. YAML keys are lowercase, no hyphens.
- **Adding a new task**: Create `src/task/<name>/` with `model.py` (`build_model`), `preprocess.py` (`get_dataloaders`), `metrics.py`. Register via config `task.name`. The `task/__init__.py` discovers it by dynamic import.
- **Git LFS**: `.pt` and `.pth` checkpoint files are tracked via Git LFS (see `.gitattributes`).
- **Accelerator extras are mutually exclusive**: only one of `cpu`/`cu124`/`rocm`/`jetson` can be installed at a time (enforced by `[tool.uv] conflicts`).
- **Dataset location**: SegFormer scripts expect `--data_root` pointing at a directory containing `train/`, `val/`, `test/` splits. Convention is `dataset/RescueNet_patches` relative to the subproject root.
- **Commits**: Short imperative subjects (e.g., `Add netqos metrics hook`). Reference issues in body (`Refs #12`). Group related changes.
