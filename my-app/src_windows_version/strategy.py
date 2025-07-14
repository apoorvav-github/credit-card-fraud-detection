import flwr as fl
from flwr.server.strategy import FedAvg
import os
import json

class SimpleAsyncFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_fit(self, rnd, results, failures):
        # In real async strategy this would be triggered per client,
        # but here we just aggregate when called
        return super().aggregate_fit(rnd, results, failures)

class LoggingStrategy(FedAvg):
    def __init__(self, results_folder="results", **kwargs):
        super().__init__(**kwargs)
        self.results_folder = results_folder
        self.client_metrics = {}

    def aggregate_evaluate(self, rnd, results, failures):
        round_metrics = {}
        for client_proxy, evaluate_res in results:
            cid = evaluate_res.metrics.get("cid", f"client_{len(round_metrics)}")
            round_metrics[cid] = {
                "loss": evaluate_res.loss,
                "auc": evaluate_res.metrics.get("auc"),
                "accuracy": evaluate_res.metrics.get("accuracy")
            }

        self.client_metrics[rnd] = round_metrics

        os.makedirs(self.results_folder, exist_ok=True)
        path = os.path.join(self.results_folder, "client_metrics.json")
        with open(path, "w") as f:
            json.dump(self.client_metrics, f, indent=4)

        print(f"\n📊 Round {rnd} Client Metrics:")
        for cid, m in round_metrics.items():
            print(f"  Client {cid}: Loss={m['loss']:.4f}, AUC={m['auc']:.4f}, Acc={m['accuracy']:.4f}")

        return super().aggregate_evaluate(rnd, results, failures)
    
def get_strategy(name="sync", fraction_fit=1.0, min_fit_clients=3, min_available_clients=3):
    if name == "sync":
        return LoggingStrategy(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            results_folder="results"
        )
    elif name == "async":
        return SimpleAsyncFedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=1,
            min_available_clients=1,
        )
    elif name == "hybrid":
        return LoggingStrategy(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            results_folder="results"
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")

    



