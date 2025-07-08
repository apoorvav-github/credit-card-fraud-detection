import flwr as fl

def get_strategy(name="sync", fraction_fit=1.0, min_fit_clients=3, min_available_clients=3):
    if name == "sync":
        return fl.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
        )
    elif name == "async":
        class AsyncStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(self, rnd, results, failures):
                # immediate averaging on each arrival
                return super().aggregate_fit(rnd, results, failures)
        return AsyncStrategy(fraction_fit=1.0, min_fit_clients=1, min_available_clients=1)
    elif name == "hybrid":
        # simple hybrid placeholder (sync strategy here)
        return fl.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")
