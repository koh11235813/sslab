# federated/server.py
import flwr as fl
def main(cfg):
    strategy = fl.server.strategy.FedAvg( # 後でFedProx/Scaffoldに差替
        min_available_clients=cfg.min_clients
    )
    fl.server.start_server(server_address=cfg.bind, strategy=strategy)

