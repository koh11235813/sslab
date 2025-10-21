# Semantic Network Project

This repository contains a modular framework for federated learning on NVIDIA Jetson devices with a focus on semantic communication.  It allows you to train lightweight models on edge devices, compress model updates or features into semantic codes, and coordinate training across a fleet of devices using Federated Averaging.  The project structure is designed to make it easy to swap out tasks (e.g. image segmentation or time‑series prediction) and adjust compression strategies based on network conditions.

## Project Structure

- **configs/** – YAML configuration files for different tasks and compression modes.
- **task/** – Implementations of task‑specific models, data preprocessing, and metrics.  Currently two example tasks are provided:
  - `disaster/` – simple image segmentation for disaster scene analysis.
  - `netqos/` – time series prediction for network quality of service (QoS) estimation.
- **semantic/** – Modules for semantic encoding/decoding and rate control of transmitted codes.
- **federated/** – Client and server implementations for the Federated Learning loop (uses [Flower](https://flower.dev/) under the hood).
- **transport/** – Prototypes for an ad‑hoc transport layer and message definitions.
- **run_single.py** – Script to train a task locally on a single Jetson device.
- **run_fed_client.py** – Run a federated learning client that trains locally and communicates updates to a server.
- **run_fed_server.py** – Launch a simple federated learning server that aggregates client updates.

To get started, install the Python dependencies and run one of the scripts with the desired configuration file.  See the docstrings in each module for more details.
