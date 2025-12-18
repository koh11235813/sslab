# Federated Learning

This package contains a minimal Flower runner plus dependency metadata shared by the `sslab` experiments. All runtime dependencies are described in `pyproject.toml`, with optional extras to match each accelerator target (Jetson, CUDA 12.4, ROCm 6.1).

## Requirements
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for dependency management and locking
- Git LFS-enabled clone of this repository if you plan to sync weight checkpoints

## Installation with uv
1. Create and activate a virtual environment:
   ```bash
   cd federated-learning
   uv venv
   source .venv/bin/activate
   ```
2. Install the base dependencies:
   ```bash
   uv sync
   ```
   The command reads `pyproject.toml` and resolves packages declared under `[project.dependencies]`. To reinstall exactly what is in `uv.lock`, run `uv sync` instead of `uv pip install`.

## Accelerator-specific extras
Only one accelerator extra can be installed at a time (conflicts are enforced under `[tool.uv]`). Pick the extra that matches your target hardware:

- **Jetson (Linux aarch64)** – prebuilt wheels hosted by NVIDIA:
  ```bash
  uv sync --extra jetson
  ```
- **CUDA 12.4 on x86_64** – standard PyTorch wheels from the cu124 index:
  ```bash
  uv sync --extra cu124
  ```
- **ROCm 6.1 on x86_64** – ROCm-enabled wheels:
  ```bash
  uv sync --extra rocm
  ```

The `[tool.uv.sources]` stanza in `pyproject.toml` points uv at the correct PyTorch index URLs, so no manual `pip` flags are required. If you ever need to rebuild a lock file for a specific accelerator, supply the `--extra` flag, for example `uv lock --extra rocm`.

## Development workflow
- Format and lint using your editor of choice; there are no additional hooks yet.
- Use `uv run python main.py` to verify the environment activates correctly.
- The main repository's runner scripts (`run_single.py`, `run_fed_server.py`, `run_fed_client.py`) can import this package once installed in editable mode (as shown above).

Refer to `AGENTS.md` in the repo root for higher-level training and evaluation guidance.
