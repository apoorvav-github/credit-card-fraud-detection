import flwr as fl
from flwr.server.strategy import FedAvg
import os
import json
import torch
import numpy as np
from model import FraudDetectionModel, set_parameters
from torch.utils.data import DataLoader

class FedProxStrategy(FedAvg):
    def __init__(self, mu=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
    # Note: For FedProx to be effective, proximal updates must be implemented client-side.

class SimpleAsyncFedAvg(FedAvg):
    def __init__(self, results_folder=None, **kwargs):
        super().__init__(**kwargs)
        self.results_folder = results_folder

    def aggregate_fit(self, rnd, results, failures):
        # Add async-specific aggregation logic here if needed
        return super().aggregate_fit(rnd, results, failures)


# This class: I have redefined to include precision, recall, and F1 score: akash
class LoggingStrategy(FedAvg):
    def __init__(self, results_folder="results", evaluate_fn=None, **kwargs):
        super().__init__(evaluate_fn=evaluate_fn, **kwargs)
        self.results_folder = results_folder
        self.client_metrics = {}
        self.server_metrics = {}

    def aggregate_evaluate(self, rnd, results, failures):
        round_metrics = {}
        # print(results)
        for _, evaluate_res in results:
            cid = evaluate_res.metrics.get("cid", f"client_{len(round_metrics)}")
            round_metrics[cid] = {
                "loss": evaluate_res.loss,
                "auc": evaluate_res.metrics.get("auc", float('nan')),
                "accuracy": evaluate_res.metrics.get("accuracy", float('nan')),
                "precision": evaluate_res.metrics.get("precision", float('nan')),
                "recall": evaluate_res.metrics.get("recall", float('nan')), 
                "f1_score": evaluate_res.metrics.get("f1_score", float('nan'))
            }

        self.client_metrics[rnd] = round_metrics

        os.makedirs(self.results_folder, exist_ok=True)
        with open(os.path.join(self.results_folder, "client_metrics.json"), "w") as f:
            json.dump(self.client_metrics, f, indent=4)

        print(f"\n📊 Round {rnd} Client Metrics:")
        for cid, m in round_metrics.items():
            print(f"  Client {cid}: Loss={m['loss']:.4f}, AUC={m['auc']:.4f}, Acc={m['accuracy']:.4f}, Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1_score']:.4f}")
        
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(rnd, results, failures)
        server_metrics = {
            "global_loss": loss_aggregated,
            "global_precision": metrics_aggregated.get("precision", float('nan')),
            "global_recall": metrics_aggregated.get("recall", float('nan')), 
            "global_f1_score": metrics_aggregated.get("f1_score", float('nan')),
            "global_accuracy": metrics_aggregated.get("accuracy", float('nan')),
            "global_auc": metrics_aggregated.get("auc", float('nan'))
            }
        if server_metrics:
            self.server_metrics[rnd] = server_metrics
            with open(os.path.join(self.results_folder, "server_metrics.json"), "w") as f:
                json.dump(self.server_metrics, f, indent=4)

        return loss_aggregated, metrics_aggregated

def weighted_average(metrics):
    """Aggregate accuracy and AUC from all clients using weighted average."""
    total = sum(num_examples for num_examples, _ in metrics)
    accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total
    precision = sum(num_examples * m["precision"] for num_examples, m in metrics) / total
    recall = sum(num_examples * m["recall"] for num_examples, m in metrics) / total
    f1 = sum(num_examples * m["f1_score"] for num_examples, m in metrics) / total
    
    auc = sum(num_examples * m["auc"] for num_examples, m in metrics) / total
    return {"accuracy": accuracy, "auc": auc,"precision": precision, "recall": recall, "f1_score": f1}

def get_strategy(
    name="sync",
    fraction_fit=1.0,
    fraction_evaluate=1.0,  # ADD THIS LINE
    min_fit_clients=3,
    min_evaluate_clients=3,  # ADD THIS LINE
    evaluate_fn=None,
    min_available_clients=3,
    central_test=None,
    results_folder="results"
):
    fedprox_mu = 0.01

    eval_fn = evaluate_fn


    if name == "sync":
        return LoggingStrategy(
            fraction_fit=1.0,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=weighted_average, # pass in the custom aggregation function
            results_folder=results_folder,
        )
    elif name == "async":
        return SimpleAsyncFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=1,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=1,
            evaluate_metrics_aggregation_fn=weighted_average,
            results_folder=results_folder,
        )
    elif name == "hybrid":
        return LoggingStrategy(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            results_folder=results_folder
        )
    elif name == "fedavg":
        return FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    elif name == "fedprox":
        return FedProxStrategy(
            mu=fedprox_mu,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            results_folder=results_folder
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")
