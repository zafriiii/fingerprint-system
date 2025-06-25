# server.py (central server for federated learning using Flower)

import flwr as fl


class NoEvalFedAvg(fl.server.strategy.FedAvg):
    def configure_evaluate(self, server_round, parameters, client_manager):
        # Disable evaluation by returning an empty list
        return []


# Start Flower server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=NoEvalFedAvg(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
    ),
)
