import argparse
import torch
import json
import os
from datetime import datetime
import flwr as fl
from flwr.common import Context
from model import FraudDetectionModel, get_parameters, set_parameters
from client_utils import FLClient
from data_utils import load_and_preprocess, split_data
from strategy import get_strategy
from torch.utils.data import DataLoader

def create_experiment_folder(base_dir="results", clients=3, strategy="sync", iid=True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"exp_{timestamp}_clients{clients}_strategy{strategy}_iid{iid}"
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
    parser.add_argument("--iid", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Create results folder and save configuration
    results_folder = create_experiment_folder(
        clients=args.clients,
        strategy=args.strategy,
        iid=args.iid
    )
    save_config(vars(args), results_folder)
    print(f"Experiment results will be saved in: {results_folder}")
    
    # Load and preprocess dataset
    X, y = load_and_preprocess(args.data_path)

    # Split data into client datasets AND central test dataset
    client_datasets, central_test = split_data(X, y, n_clients=args.clients, iid=args.iid)

    # For each client dataset, split further into train and test subsets (e.g., 80/20 split)
    client_train_loaders = {}
    client_test_loaders = {}
    for i, dataset in enumerate(client_datasets):
        # Convert TensorDataset to arrays for splitting
        X_client = dataset.tensors[0].numpy()
        y_client = dataset.tensors[1].numpy()

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=0.2, random_state=42, stratify=y_client
        )

        train_ds = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        test_ds = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

        client_train_loaders[i] = DataLoader(train_ds, batch_size=32, shuffle=True)
        client_test_loaders[i] = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Get FL strategy
    strategy = get_strategy(
        name=args.strategy,
        min_fit_clients=args.clients,
        min_available_clients=args.clients
    )
    
    def client_fn(cid: str):
        cid_int = int(cid)
        train_loader = client_train_loaders[cid_int]
        test_loader = client_test_loaders[cid_int]
        return FLClient(cid=cid, train_loader=train_loader, test_loader=test_loader)


    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
