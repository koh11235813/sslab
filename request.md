# P2P Federated Learning

## Create a new branch

First, create a new branch for your work. Choose a clear and descriptive name.

```sh
git checkout -b feature/p2p_federated_learning
```

Goal

On this branch, we will add functionality for two Jetson devices to learn by exchanging model updates peer-to-peer, without going through a central server. The existing code uses Flower for centralized FedAvg; here, we implement P2P mode.

## Specific Tasks

1. Create a new script: run_peer_training.py

Add a new entry script named run_peer_training.py directly under the semantic-net project root (not under src). Its role is as follows:

Accepts --config (YAML configuration file) and --peers (comma-separated list of ip:port) via argparse.

Reads the configuration file to generate the model and data loader (reuses the existing task.load_task).

Using AdHocTransport from semantic-net/transport/ad_hoc.py, implement the following flow: open a local port, train your own model during each communication round → serialize model parameters → send to all peers → average updates received from peers → update the local model.

Reduce communication overhead by using sparsify_topk and quantize_tensor from semantic/encoder.py for transmitted/received data, depending on the mode specified in cfg[‘semantic’] (none / gradients_topk / weights_quantize).

When sending, serialize numpy arrays using pickle or msgpack; on the receiving end, deserialize them back to torch.tensor.

2. Peer integration logic referencing federated/client.py

For the model update logic within run_peer_training.py, refer to JetsonClient.fit in federated/client.py. However, do not use the Flower client; instead, write your own loop.

For example, after training for one epoch, retrieve the parameter lists for all layers, convert each tensor to numpy, and send them after quantizing and sparsifying as needed.

The receiver waits until updates for the same round arrive from all peers. It then dequantizes the received data, averages it with its own parameters, and proceeds to the next round.

3. Extending the Configuration File

Add a new section named `p2p` to `configs/base.yaml`. This section allows specifying the default local port (e.g., 5000) and listen timeout.

`run_peer_training.py` should read this configuration section and override values as needed.

4. Update README

Add instructions for using peer-to-peer learning mode to README.md. Include command examples like the following:

```sh
python run_peer_training.py --config configs/task_disaster.yaml --peers 192.168.0.2:5000,192.168.0.3:5000
```

5. Handling Added Dependencies

When introducing a new package, add it to the [project.dependencies] section of pyproject.toml and update uv.lock. However, avoid adding packages if the functionality can be achieved using only the existing Python standard library.

6. Simple Test

You may implement a simple test by running run_peer_training.py in two local terminals (or on separate localhost ports) to verify that model parameters are actually being exchanged (if impractical, a description alone is acceptable).

## Final Verification

Once complete, verify changes by executing the following on the feature/p2p_federated_learning branch and commit.

```sh
git add .
git commit -m "Implement peer-to-peer federated training between Jetson devices"
```

If you have an environment where you can push `git pushr` as needed, please include that information as well.
