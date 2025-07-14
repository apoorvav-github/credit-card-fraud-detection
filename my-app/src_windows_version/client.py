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
    parser.add_argument("--iid", action="store_true")
    args = parser.parse_args()

    # Load and preprocess data
    X, y = load_and_preprocess(args.data_path)

    # Split data among clients
    client_datasets, _ = split_data(X, y, n_clients=args.clients, iid=args.iid)

    # Get client-specific datasets
    train_ds = client_datasets[args.cid]
    test_ds = train_ds  # Or implement a separate test split if needed

    # Wrap datasets with DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Flower client
    client = FLClient(str(args.cid), train_loader, test_loader, device)

    # Start Flower client
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()
