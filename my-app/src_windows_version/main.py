import argparse
import torch
import numpy as np
import json
import os
from datetime import datetime
import flwr as fl
from model import FraudDetectionModel, get_parameters, set_parameters
from client_utils import FLClient
from flwr.common import Context
from data_utils import load_and_preprocess, split_data
from strategy import get_strategy
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def create_experiment_folder(base_dir="results", clients=3, strategy="sync", distribution="iid"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"exp_{timestamp}_clients{clients}_strategy{strategy}_dist-{distribution}"
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def save_config(config: dict, folder_path: str):
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

class LoggingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, results_folder, **kwargs):
        super().__init__(**kwargs)
        self.results_folder = results_folder
        self.metrics = {}

    def aggregate_evaluate(self, rnd, results, failures):
        res = super().aggregate_evaluate(rnd, results, failures)
        if res and res.metrics:
            self.metrics[rnd] = res.metrics
            # Save metrics to a JSON file
            metrics_path = os.path.join(self.results_folder, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=4)
        return res

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--strategy", type=str, choices=["sync", "async", "hybrid"], default="sync")
    # parser.add_argument("--iid", action="store_true")
    parser.add_argument("--distribution", type=str, default="iid",
        choices=["iid", "non-iid-weak", "non-iid-medium", "non-iid-strong"],
        help="Data distribution type across clients")

    return parser.parse_args()


# ✅ This is the new main() required by flwr run
def main():

    args = parse_args()

    results_folder = create_experiment_folder(
        clients=args.clients,
        strategy=args.strategy,
        distribution=args.distribution
    )
    save_config(vars(args), results_folder)
    print(f"Experiment results will be saved in: {results_folder}")

    X, y = load_and_preprocess(args.data_path)
    client_datasets, central_test = split_data(X, y, n_clients=args.clients, distribution=args.distribution)

    client_train_loaders = {}
    client_test_loaders = {}

    for i, dataset in enumerate(client_datasets):
        X_client = dataset.tensors[0].numpy()
        y_client = dataset.tensors[1].numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=0.2, random_state=42, stratify=y_client
        )

        train_ds = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        test_ds = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

        client_train_loaders[i] = DataLoader(train_ds, batch_size=32, shuffle=True)
        client_test_loaders[i] = DataLoader(test_ds, batch_size=64, shuffle=False)

    strategy = get_strategy(
        name=args.strategy,
        min_fit_clients=args.clients,
        min_available_clients=args.clients,
        # evaluate_fn=evaluate_fn,
        results_folder="results"
    )

    def client_fn(cid: str):
        cid_int = int(cid)
        train_loader = client_train_loaders[cid_int]
        test_loader = client_test_loaders[cid_int]
        return FLClient(cid=cid, train_loader=train_loader, test_loader=test_loader)

    # def client_fn(ctx: Context):
    #     cid_int = int(ctx.client_id)
    #     train_loader = client_train_loaders[cid_int]
    #     test_loader = client_test_loaders[cid_int]
    #     return FLClient(cid=ctx.client_id, train_loader=train_loader, test_loader=test_loader)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy
    )
if __name__ == "__main__":
    main()