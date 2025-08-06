import argparse
import torch
import numpy as np
import json
import os
from datetime import datetime
import flwr as fl
from model import FraudDetectionModel, get_parameters, set_parameters
from client_utils import FLClient, train, evaluate_model
from flwr.common import Context
from data_utils import load_and_preprocess, split_data
from strategy import get_strategy, weighted_average
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def create_experiment_folder(base_dir="results", clients=3, strategy="sync", distribution="iid", isolated=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "isolated" if isolated else "federated"
    folder_name = f"exp_{timestamp}_{mode}_clients{clients}_strategy{strategy}_dist-{distribution}"
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
    parser.add_argument("--strategy", type=str, choices=["sync", "async", "hybrid", "fedavg", "fedprox"], default="sync")
    parser.add_argument("--isolated", action="store_true", 
        help="Run isolated training instead of federated learning")
    # parser.add_argument("--iid", action="store_true")
    parser.add_argument("--distribution", type=str, default="iid",
        choices=["iid", "non-iid-weak", "non-iid-medium", "non-iid-strong"],
        help="Data distribution type across clients")

    return parser.parse_args()


def run_isolated_training(args, client_train_loaders, client_test_loaders, results_folder):
    """
    Run isolated training for each client separately without federated learning
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Format: round -> client -> metrics
    round_based_metrics = {}
    
    # Initialize models for all clients
    client_models = {}
    for client_id in range(args.clients):
        client_models[client_id] = FraudDetectionModel()
    
    # Train for specified rounds
    for round_num in range(1, args.rounds + 1):
        print(f"\n=== Round {round_num}/{args.rounds} ===")
        round_based_metrics[str(round_num)] = {}
        
        for client_id in range(args.clients):
            print(f"Training Client {client_id}")
            
            model = client_models[client_id]
            train_loader = client_train_loaders[client_id]
            test_loader = client_test_loaders[client_id]
            
            # Train the model for one epoch
            train(model, train_loader, epochs=1, device=device)
            
            # Evaluate the model
            test_loss, test_metrics = evaluate_model(model, test_loader, device)
            
            # Store metrics in the desired format
            round_based_metrics[str(round_num)][str(client_id)] = {
                "loss": test_loss,
                "auc": test_metrics["auc"],
                "accuracy": test_metrics["accuracy"]
            }
            
            print(f"  Client {client_id} - Loss: {test_loss:.4f}, AUC: {test_metrics['auc']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Save metrics in the desired format (round -> client -> metrics)
    client_metrics_path = os.path.join(results_folder, "client_metrics.json")
    with open(client_metrics_path, "w") as f:
        json.dump(round_based_metrics, f, indent=4)
    
    # Calculate final statistics for summary display
    final_round_data = round_based_metrics[str(args.rounds)]
    final_aucs = [metrics["auc"] for metrics in final_round_data.values()]
    final_accuracies = [metrics["accuracy"] for metrics in final_round_data.values()]
    avg_auc = sum(final_aucs) / len(final_aucs)
    avg_accuracy = sum(final_accuracies) / len(final_accuracies)
    
    print(f"\n=== Isolated Training Complete ===")
    print(f"Results saved in: {results_folder}")
    print(f"Final Average AUC: {avg_auc:.4f}")
    print(f"Final Average Accuracy: {avg_accuracy:.4f}")
    print(f"Files saved: config.json, client_metrics.json")


# ✅ This is the new main() required by flwr run
def main():

    args = parse_args()

    results_folder = create_experiment_folder(
        clients=args.clients,
        strategy=args.strategy,
        distribution=args.distribution,
        isolated=args.isolated
    )
    save_config(vars(args), results_folder)
    print(f"Experiment results will be saved in: {results_folder}")

    X, y = load_and_preprocess(args.data_path)
    client_datasets, central_test = split_data(X, y, n_clients=args.clients, distribution=args.distribution)

    client_train_loaders = {}
    client_test_loaders = {}

    # def wrapped_evaluate_fn(server_round, parameters):
    #     return evaluate_fn(args.rounds, parameters, central_test, results_folder="results")

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

    # Choose between isolated and federated training
    if args.isolated:
        print("=== Running Isolated Training Mode===")
        run_isolated_training(args, client_train_loaders, client_test_loaders, results_folder)
    else:
        print("=== Running Federated Learning Mode===")
        strategy = get_strategy(
            name=args.strategy,
            min_fit_clients=args.clients,
            min_available_clients=args.clients,
            evaluate_fn=weighted_average,
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