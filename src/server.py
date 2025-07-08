import argparse
import flwr as fl
from strategy_utils import get_strategy

def main():
    parser = argparse.ArgumentParser("FL Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--strategy", choices=["sync","async","hybrid"], default="sync")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    strategy = get_strategy(args.strategy, min_fit_clients=args.clients, min_available_clients=args.clients)

    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    print(f"Server running on {args.host}:{args.port} ({args.strategy})")

if __name__ == "__main__":
    main()
