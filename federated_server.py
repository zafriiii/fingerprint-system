# This file is the central server for federated learning
import flwr as fl


# This disables evaluation on the server
class NoEvalFedAvg(fl.server.strategy.FedAvg):
    def configure_evaluate(self, server_round, parameters, client_manager):
        # Do not run evaluation, just return empty list
        return []


# Start the Flower server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=NoEvalFedAvg(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
    ),
)
