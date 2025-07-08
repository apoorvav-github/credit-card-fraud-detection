import flwr as fl
from flwr.server.strategy import FedAvg

class SimpleAsyncFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_fit(self, rnd, results, failures):
        # In real async strategy this would be triggered per client,
        # but here we just aggregate when called
        return super().aggregate_fit(rnd, results, failures)

def get_strategy(name="sync", fraction_fit=1.0, min_fit_clients=3, min_available_clients=3):
    if name == "sync":
        return FedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
        )
    elif name == "async":
        return SimpleAsyncFedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=1,
            min_available_clients=1,
        )
    elif name == "hybrid":
        return FedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")
