import flwr as fl
from flwr.server.strategy import FedAvg
import os
import json
import torch
import numpy as np
from model import FraudDetectionModel,set_parameters
from torch.utils.data import DataLoader

class SimpleAsyncFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_fit(self, rnd, results, failures):
        # In real async strategy this would be triggered per client,
        # but here we just aggregate when called
        return super().aggregate_fit(rnd, results, failures)

class LoggingStrategy(FedAvg):
    def __init__(self, results_folder="results",evaluate_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.results_folder = results_folder
        self.client_metrics = {}

        # super().__init__(evaluate_fn=evaluate_fn, **kwargs)
        # self.results_folder = results_folder
        # self.client_metrics = {}

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
"""
def evaluate_fn(server_round, parameters, config,results_folder="results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FraudDetectionModel()
    set_parameters(model, parameters)
    model.eval()
    model.to(device)

    loss_fn = torch.nn.BCELoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in DataLoader(central_test, batch_size=64):
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb)

            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_preds)

    print(f" [Server Evaluation] Round {server_round}: Acc={acc:.4f}, AUC={auc:.4f}, Loss={avg_loss:.4f}")

    # Save server evaluation to results folder
    server_eval_path = os.path.join(results_folder, "server_eval.json")
    if not os.path.exists(server_eval_path):
        with open(server_eval_path, "w") as f:
            json.dump({}, f)

    with open(server_eval_path, "r+") as f:
        data = json.load(f)
        data[f"round_{server_round}"] = {
            "loss": avg_loss,
            "accuracy": acc,
            "auc": auc
        }
        f.seek(0)
        json.dump(data, f, indent=4)

    return avg_loss, {"accuracy": acc, "auc": auc}
"""
def get_strategy(name="sync", fraction_fit=1.0, min_fit_clients=3, min_available_clients=3, evaluate_fn=None, results_folder="results"):
    
    if name == "sync":
        return LoggingStrategy(
            fraction_fit=1.0,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            # evaluate_fn=evaluate_fn,
            results_folder=results_folder,
        )
    elif name == "async":
        return SimpleAsyncFedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=1,
            min_available_clients=1,
            # evaluate_fn=evaluate_fn,
            results_folder=results_folder,
        )
    elif name == "hybrid":
        return LoggingStrategy(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            # evaluate_fn=evaluate_fn,
            results_folder="results"
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")

    



