import argparse
import torch
import flwr as fl
from torch.utils.data import DataLoader
from data_utils import load_and_preprocess, split_data
from client_utils import FLClient

def main():
    parser = argparse.ArgumentParser("FL Client")
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--clients", type=int, default=3)
    # parser.add_argument("--iid", action="store_true")
    parser.add_argument("--distribution", type=str, default="iid",
    choices=["iid", "non-iid-weak", "non-iid-medium", "non-iid-strong"],
    help="Data distribution type across clients")
    
    args = parser.parse_args()

    # RECOMMENDED: Validate client ID range
    assert 0 <= args.cid < args.clients, (
        f"Client ID {args.cid} must be in the range [0, {args.clients - 1}]")
    
    # Load and preprocess data
    X, y = load_and_preprocess(args.data_path)

    # Split data among clients
    client_datasets, central_test = split_data(X, y, n_clients=args.clients, distribution=args.distribution)

    # Get client-specific datasets
    train_ds = client_datasets[args.cid]
    test_ds = central_test

    # Wrap datasets with DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"   Client {args.cid} initialized.")
    print(f"   Distribution: {args.distribution}, Total clients: {args.clients}")
    print(f"   Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # Initialize Flower client
    client = FLClient(str(args.cid), train_loader, test_loader, device)
    # Start Flower client
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()
