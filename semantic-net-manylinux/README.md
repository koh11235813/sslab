# semantic-net-manylinux

Federated semantic communication experiments that combine lightweight semantic encoders with Flower-based training loops. The repository targets Python 3.10–3.11 on Linux/x86_64 (Jetson devices work when CUDA is available) and ships extras for CPU, CUDA 12.4, and ROCm 6.1 builds via `uv`.

## Repository Layout
- `src/run_single.py`, `run_fed_client.py`, `run_fed_server.py` – entry points for single-device smoke tests and Flower client/server workflows.
- `src/semantic/` – sparsification, quantization, and the simple `RateController` used to throttle payload sizes (`semantic.encoder`, `decoder`, `rate_controller`).
- `src/federated/` – the `JetsonClient` Flower client wrapper plus a FedAvg server helper.
- `src/task/` – task packages. Each task contains `model.py`, `preprocess.py`, and `metrics.py`. `disaster` uses a toy segmentation network, `netqos` implements an LSTM forecaster.
- `configs/` – YAML presets consumed by the runners (`configs/task_disaster.yaml` is the minimal template).
- `src/transport/` – experimental UDP transport scaffolding (`AdHocTransport`) and dataclasses describing semantic packets.
- `scripts/` – utility scripts such as `scripts/smoke.py` for verifying the PyTorch install.
- `src/dataset/` – helpers for working with the RescueNet segmentation patches plus resizing scripts.
- `train_segformer_b*.py` and the `checkpoints_segformer_b*` folders – standalone SegFormer training utilities for heavier segmentation experiments.

## Requirements & Setup
1. Install system dependencies, then create a virtual environment (or reuse one from `uv sync`):
   ```sh
   python3 -m venv .venv && source .venv/bin/activate
   pip install uv
   ```
2. Sync dependencies with the desired backend (default is CPU). Extras match the `pyproject.toml` specifications:
   ```sh
   # CPU backend
   uv sync --extra cpu

   # CUDA 12.4 backend (requires matching NVIDIA drivers)
   uv sync --extra cu124

   # ROCm 6.1 backend (Linux/x86_64)
   uv sync --extra rocm
   ```
3. Confirm the environment is functional:
   ```sh
   uv run python scripts/smoke.py
   # or: uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

## Quick Start
### Single-Device Training
Both bundled tasks operate on synthetic data so they run anywhere. Override batch size, learning rate, device, and checkpoint paths via flags:
```sh
python src/run_single.py --task disaster --epochs 1
python src/run_single.py --task netqos --epochs 5 --batch_size 32 --lr 5e-4 --device cpu
python src/run_single.py --task disaster --epochs 3 --save-model --output-path checkpoints/disaster_demo.pth
```
For single-device smoke tests, setting `semantic.mode` to `gradients_topk` sparsifies gradients before the optimizer step. The federated client also honors `semantic.mode=weights_quantize` to return quantized model updates.

### Federated Round Trip
Use Flower to simulate a server and client on your workstation:
```sh
# Terminal 1
python src/run_fed_server.py --port 8080 --rounds 3

# Terminal 2
python src/run_fed_client.py --config configs/task_disaster.yaml --server localhost:8080
```
The client loads task components via `task.load_task`, trains for the configured `training.local_epochs`, optionally applies semantic compression, and then reports validation loss. Multiple clients can attach to the same server with different configs.

### Docker Compose
Containerized workflows set up the same runners under `/opt/semantic`:
```sh
# Build with UV_EXTRA=cu124 or UV_EXTRA=rocm for GPU images
docker compose build

# Quick single-device run
docker compose run --rm run-single

# Launch Flower server + attach a client
docker compose up client

# Interactive shell with uv environment pre-synced
docker compose run --rm workspace bash
```

## Configuration
All runners accept a configuration dictionary with three top-level keys:
- `task`: contains `name` (`disaster` or `netqos`) plus task-specific args such as `num_classes`, `input_size`, or model hyperparameters.
- `training`: includes `local_epochs`, `batch_size`, `lr`, and synthetic dataset controls (`image_size`, `seq_length`, `num_samples`, `val_split`).
- `semantic`: toggles compression strategies. Supported keys are `mode` (`none`, `gradients_topk`, `weights_quantize`), `topk` (fraction retained when sparsifying), and `bit_width` (1–8 bits when quantizing).

Example (`configs/task_disaster.yaml`):
```yaml
task:
  name: disaster
  num_classes: 2

training:
  local_epochs: 1
  batch_size: 8
  image_size: 128
  num_samples: 60
  val_split: 0.2
  lr: 0.001

semantic:
  mode: none
```
Start from this preset, copy it when defining new tasks, and keep overrides minimal so diffs stay readable.

## Semantic Compression & Rate Control
- `semantic.encoder.sparsify_topk` zeroes out all but the highest-magnitude parameters/gradients.
- `semantic.encoder.quantize_tensor` + `semantic.decoder.dequantize_tensor` convert tensors to low-precision affine representations.
- `semantic.rate_controller.RateController` adapts quantization bit width based on observed RTT and loss; wire it into your client loop to auto-tune `semantic.bit_width`.

## Extending Tasks
To add a new task (e.g., a custom dataset):
1. Create `src/task/<task_name>/` with `model.py`, `preprocess.py`, and `metrics.py`. Follow the disaster/netqos modules for expected function signatures (`build_model`, `get_dataloaders`, metric helpers).
2. Add or clone a YAML config under `configs/` and set `task.name` to your new identifier.
3. Point `run_single.py` or `run_fed_client.py` at the new config and iterate.

The `src/dataset/` utilities (RescueNet patch loader, resizing helpers, and sample data) are available if you want to swap the dummy datasets for real segmentation data.

## Transport & Measurement Utilities
Early-stage transport experiments live in `src/transport/`. `AdHocTransport` demonstrates how to send quantized payloads over UDP, while `transport.proto` defines `ModelDelta` and `FeaturePacket` dataclasses for custom serialization. `src/measure_latency_jetson.py` helps capture link metrics for feeding into the rate controller, and the SegFormer training/checkpoint scripts provide stronger disaster models if you need to benchmark semantic compression against higher-quality baselines.
