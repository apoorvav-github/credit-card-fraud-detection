import argparse
import flwr as fl
from data_utils import load_and_preprocess, split_data
from model_utils import MLP
from client_utils import FLClient
from strategy_utils import get_strategy

def main():
    parser = argparse.ArgumentParser("FL Local Simulation")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--strategy", choices=["sync","async","hybrid"], default="sync")
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--clients", type=int, default=3)
    args = parser.parse_args()

    X, y = load_and_preprocess(args.data_path)
    client_datasets, central_test = split_data(X, y, n_clients=args.clients, iid=args.iid)

    def client_fn(cid: str):
        idx = int(cid)
        train_ds = client_datasets[idx]
        test_ds = central_test
        return FLClient(cid, train_ds, test_ds)

    strategy = get_strategy(args.strategy, min_fit_clients=args.clients, min_available_clients=args.clients)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
