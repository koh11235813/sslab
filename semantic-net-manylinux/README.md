# semantic-net-manylinux

Federated semantic communication experiments packaged for x86_64 Linux only. Runtime bindings are selected at install time using extras for CPU, CUDA 12.8, or ROCm 6.4.

## Installation
```sh
# CPU backend
uv sync --extra cpu

# CUDA 12.8 backend (requires compatible NVIDIA drivers)
uv sync --extra cu128

# ROCm 6.4 backend (requires ROCm runtime/toolchain)
uv sync --extra rocm
```

## Quick Check
After syncing with the desired extra, verify PyTorch by running:
```sh
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
