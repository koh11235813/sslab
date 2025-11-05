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

## Docker Compose
Build and run the project inside containers using the bundled Docker configuration:

```sh
# Build the CPU image (set UV_EXTRA=cu128 or UV_EXTRA=rocm for GPU backends)
docker compose build

# Run a quick single-device training loop
docker compose run --rm run-single

# Start a Flower server (exposes port 8080) and attach a client
docker compose up client
```

For interactive experiments, drop into a shell that has the synced environment:
```sh
docker compose run --rm workspace bash
```
