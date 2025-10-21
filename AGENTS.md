# Repository Guidelines

## Project Structure & Module Organization
Core logic lives in `semantic/` (encoders, decoders, rate control) and `federated/` (Flower client/server wrappers). Task-specific models, preprocessing, and metrics live under `task/`; the existing `disaster/` and `netqos/` folders show the expected layout when introducing a new task. YAML presets in `configs/` capture training, semantic compression, and transport parameters used by the runner scripts. Experimental transport scaffolding is staged in `transport/`, while the top-level scripts (`run_single.py`, `run_fed_client.py`, `run_fed_server.py`) wire everything together.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate` to isolate dependencies (Torch, Flower, PyYAML, numpy).
- `python3 run_single.py --task disaster --epochs 1` provides a quick local smoke test of task wiring.
- `python3 run_fed_server.py --port 8080 --rounds 3` starts a minimal Flower server.
- `python3 run_fed_client.py --config configs/task_disaster.yaml --server localhost:8080` joins a client using the selected config.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and descriptive snake_case names for functions, modules, and configs. Keep modules type-annotated as in `federated/client.py`, and prefer dataclass-like dictionaries for configuration. Add docstrings describing purpose and IO; mirror existing module headers when adding new files. Treat YAML keys as lower-case, hyphen-free identifiers for consistency across configs and scripts.

## Testing Guidelines
There is no automated suite yet; add targeted unit tests under a new `tests/` package using `pytest` and lightweight fixtures from `task/`. Exercise semantic compression helpers with deterministic tensors and pin random seeds for reproducibility. Before submitting, run the relevant script (`run_single.py` or the federated pair) with abbreviate epochs to confirm end-to-end behavior.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects (e.g., `Add netqos metrics hook`) followed by wrapped detail when needed.
- Group related changes; avoid mixing task definitions with transport tweaks in one commit.
- Reference GitHub issues in the body (`Refs #12`) and note config versions touched.
- PRs should summarize motivation, outline test evidence, and mention any new configs or scripts to run.
- Include screenshots or log snippets when changing runtime output or CLI UX.

## Configuration Tips
When cloning or extending configs, start from `configs/base.yaml` and override only task-specific sections to keep diffs small. Document new parameters inline with YAML comments and ensure runner examples in `README.md` remain valid.
