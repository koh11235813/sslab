"""A simple federated learning client implementation.

This module defines a ``JetsonClient`` class compatible with the
Flower federated learning framework.  It trains a model locally
using data loaders provided by the task and returns updated model
parameters to the server.  The client also supports optional
sparsification and quantization of the model weights via the
``semantic.encoder`` utilities.  The intent is to minimize
communication overhead on bandwidth‑constrained devices such as
NVIDIA Jetson platforms.

If Flower is not installed, importing this module will raise an
ImportError.  Installing Flower can be done via pip::

    pip install flwr

"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch

try:
    import flwr as fl  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The federated client requires Flower (flwr) to be installed. "
        "Install it via `pip install flwr`."
    ) from exc

from task import load_task
from semantic.encoder import sparsify_topk, quantize_tensor


class JetsonClient(fl.client.NumPyClient):
    """Federated client running on a Jetson device.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing task and training
        parameters.  See README for expected keys.
    device : torch.device, optional
        Device on which to perform computations.  Defaults to CPU if
        CUDA is unavailable.
    """

    def __init__(self, cfg: Dict[str, object], device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        components = load_task(cfg)
        self.model = components["model"].to(self.device)
        self.loaders = components["loaders"]
        self.loss_fn = torch.nn.MSELoss() if cfg.get("task", {}).get("name") == "netqos" else torch.nn.CrossEntropyLoss()
        self.semantic_mode = cfg.get("semantic", {}).get("mode", "none")
        self.topk = cfg.get("semantic", {}).get("topk", 0.1)
        self.bit_width = cfg.get("semantic", {}).get("bit_width", 8)

    # Flower client API methods
    def get_parameters(self, config: Dict[str, object]) -> List[np.ndarray]:  # type: ignore
        """Return the current model parameters as a list of NumPy arrays."""
        return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:  # type: ignore
        """Set model parameters from a list of NumPy arrays provided by the server."""
        state_dict = self.model.state_dict()
        for (name, tensor), np_array in zip(state_dict.items(), parameters):
            state_dict[name] = torch.tensor(np_array).to(tensor.device).type_as(tensor)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, object]) -> Tuple[List[np.ndarray], int, Dict]:  # type: ignore
        """Train the model for a number of local epochs and return updated parameters.

        This method receives the global model parameters from the server,
        updates the local model, runs local training and then returns the
        updated parameters.  Optionally applies sparsification or
        quantization before returning to reduce communication overhead.
        """
        import numpy as np  # Local import to avoid global dependency if not needed

        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.get("training", {}).get("lr", 1e-3))
        local_epochs = self.cfg.get("training", {}).get("local_epochs", 1)
        for epoch in range(local_epochs):
            for batch in self.loaders["train"]:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                # For segmentation tasks, reshape targets appropriately
                if self.cfg.get("task", {}).get("name") == "disaster":
                    loss = self.loss_fn(outputs, targets.long())
                else:
                    loss = self.loss_fn(outputs.squeeze(), targets.squeeze())
                loss.backward()
                if self.semantic_mode == "gradients_topk":
                    # Sparsify gradients in place
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data = sparsify_topk(p.grad.data, k=self.topk)
                optimizer.step()

        # Prepare updated parameters for return.
        params = [v.detach().cpu() for v in self.model.state_dict().values()]
        if self.semantic_mode == "weights_quantize":
            quantized_params: List[np.ndarray] = []
            for p in params:
                q_tensor, scale, zero = quantize_tensor(p, bit_width=self.bit_width)
                # Pack scale and zero into the dtype to keep the interface simple.
                # This is a toy example; in practice you would transmit scale and zero separately.
                q_tensor = q_tensor.to(torch.int32)
                quantized_params.append(q_tensor.numpy())
            params_out = quantized_params
        else:
            params_out = [p.numpy() for p in params]
        num_examples = len(self.loaders["train"].dataset)
        return params_out, num_examples, {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, object]) -> Tuple[float, int, Dict]:  # type: ignore
        """Evaluate the model on the validation data and return the loss."""
        import numpy as np  # Local import

        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for inputs, targets in self.loaders["val"]:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                if self.cfg.get("task", {}).get("name") == "disaster":
                    loss = self.loss_fn(outputs, targets.long())
                else:
                    loss = self.loss_fn(outputs.squeeze(), targets.squeeze())
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                count += batch_size
        avg_loss = total_loss / count if count > 0 else 0.0
        return avg_loss, count, {}


def start_client(cfg: Dict[str, object], server_address: str) -> None:
    """Helper function to start the federated client with the given config."""
    client = JetsonClient(cfg)
    fl.client.start_numpy_client(server_address, client=client)
