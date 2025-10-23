# Semantic Network Project

Federated learning on bandwidth-constrained edge devices benefits from exchanging compact, task-aware representations instead of raw gradients. This repository provides a lightweight playground for those ideas: it couples task-specific PyTorch models with semantic compression utilities and minimal [Flower](https://flower.dev/) client/server wrappers so you can experiment on NVIDIA Jetson-class hardware or any CUDA-capable machine.

For a Japanese translation of this guide, see [README-jp.md](README-jp.md).

## Repository Tour

- `configs/` – YAML presets for single-device and federated runs; start from `base.yaml` when cloning settings.
- `src/federated/` – Thin Flower wrappers that expose a NumPy client and FedAvg server tuned for this project.
- `src/semantic/` – Encoders, decoders, and rate control helpers implementing sparsification and quantization flows.
- `src/task/` – Self-contained tasks (models, preprocessing, metrics); `disaster/` covers segmentation, `netqos/` covers time-series forecasting.
- `src/transport/` – Experimental messaging scaffolding for future custom transports.
- `src/run_single.py` / `src/run_fed_client.py` / `src/run_fed_server.py` – Entry points that wire configs, tasks, and semantic compression together.

## Getting Started with `uv`

[`uv`](https://github.com/astral-sh/uv) manages dependencies and virtual environments for this project. The steps below create an isolated environment, install everything declared in `pyproject.toml`, and keep lock-step with the repo.

1. **Sync dependencies**

   ```bash
   uv sync
   ```

   This resolves the environment described in `pyproject.toml` into `.venv/`.

   For Jetson or other NVIDIA L4T systems, include the Jetson-specific wheels packaged with this repository:

   ```bash
   uv sync --extra jetson
   ```

2. **Use the environment**

   - Run scripts without manual activation:

     ```bash
     uv run python src/run_single.py --task disaster --epochs 1
     ```

   - Or activate the environment once per shell:

     ```bash
     source .venv/bin/activate
     ```

3. **Update dependencies**

   ```bash
   uv add <package>
   uv remove <package>
   ```

   Re-run `uv sync` when the dependency list changes.

## Docker Image (JetPack)

The provided `Dockerfile` targets NVIDIA Jetson devices by starting from `nvcr.io/nvidia/l4t-base:r36.4.0`. It installs the custom PyTorch 2.5 wheel published by NVIDIA and pairs it with the matching torchvision wheel stored under `wheels/`.

1. **Build**

   ```bash
   docker build -t semantic-net:jp36 .
   ```

   > Note: To target a different JetPack release, update the `jetson` extra in `pyproject.toml`, add the matching torchvision wheel under `wheels/`, and regenerate `uv.lock` before rebuilding.

   Supply new wheel versions if NVIDIA publishes an update:

   ```bash
   uv lock  # regenerate the lock file after updating pyproject/wheels
   docker build -t semantic-net:jp36 .
   ```

2. **Run**

   ```bash
   docker run --rm -it --runtime nvidia --network host semantic-net:jp36 bash
   ```

   The container drops you into `/workspace/semantic-net` with a virtual environment already on `PATH`. Use the installed console scripts (`semantic-run-single`, `semantic-run-fed-client`, `semantic-run-fed-server`) directly.

## Docker Compose

The repository also ships with a convenience `docker-compose.yml` that wraps the same image for interactive shells, Flower servers, and clients.

- Launch an interactive shell in the container (Jetson host) while keeping code mounted from the host machine:

  ```bash
  docker compose --profile cli run --rm shell
  ```

- Bring up a Flower server and client pair (scale clients as needed):

  ```bash
  docker compose --profile gpu up --build server
  docker compose --profile gpu up --build --scale client=2 client
  ```

  Tune runtime parameters with environment variables, e.g. `SERVER_ROUNDS=5 CLIENT_CONFIG=configs/task_netqos.yaml docker compose --profile gpu up client`.

## Usage Examples

### Single-device experiments

Run a quick smoke test of the segmentation task:

```bash
uv run python src/run_single.py --task disaster --epochs 1
```

Switch to the QoS forecasting task and override batch size and learning rate:

```bash
uv run python src/run_single.py --task netqos --epochs 3 --batch_size 32 --lr 5e-4
```

Persist trained weights after the loop finishes by adding the save flag (use `.pt` or `.pth`):

```bash
uv run python src/run_single.py --task disaster --epochs 1 --save-model --output-path checkpoints/disaster.pth
```

### Federated simulations

1. Start the server (terminal 1):

   ```bash
   uv run python src/run_fed_server.py --port 8080 --rounds 3
   ```

2. Launch a client (terminal 2):

   ```bash
   uv run python src/run_fed_client.py \
       --config configs/task_disaster.yaml \
       --server localhost:8080
   ```

   Modify `configs/task_disaster.yaml` or `configs/task_netqos.yaml` to change semantic compression modes, optimizer settings, or dataset sizes.

### Explore semantic compression

- Tweak `semantic.encoder` to prototype new quantizers or sparsification schemes.
- Adapt `semantic/rate_controller.py` if you want to enforce bitrate schedules across rounds.
- Use `transport/` as a sandbox for non-Flower communication backends or custom message formats.

## Development Workflow

- Use `uv run pytest` to execute tests once you add them under `tests/`.
- Pin reproducibility seeds inside task modules; these synthetic datasets use deterministic seeds when provided.
- When adding a new task, mirror the layout under `task/` (preprocess, model, metrics) and extend `task/__init__.py` so the loaders can discover it.
- Commit related configuration tweaks alongside code changes and document runnable presets in this file or the config comments.

## License

MIT License – see `../LICENSE` for details.
