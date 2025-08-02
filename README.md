# Credit Card Fraud Detection (Federated Learning)

This project implements a federated learning approach for credit card fraud detection using [Flower](https://flower.dev/) and PyTorch.

## Features

- Federated learning simulation with configurable number of clients
- Supports synchronous, asynchronous, and hybrid FL strategies
- Experiment tracking with results and metrics saved to disk
- Data preprocessing and splitting for IID/Non-IID scenarios

## Project Structure

```
credit-card-fraud-detection/
├── src/
│   ├── main.py
│   ├── model.py
│   ├── client_utils.py
│   ├── data_utils.py
│   └── strategy.py
├── data/
│   └── raw/creditcard.csv
├── results/
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your data

Place your dataset in a suitable location. The script expects a data path via `--data-path`.

### 3. Run the simulation

```bash
python .\src\main.py --data-path=<path_to_your_data> --clients 3 --rounds 2 --strategy sync --distribution non-iid-weak
```

- `--clients`: Number of federated clients
- `--rounds`: Number of FL rounds
- `--strategy`: FL strategy (`sync`, `async`, `hybrid`)
- `--distribution`: Use IID/Non-IID data split (iid", "non-iid-weak", "non-iid-medium", "non-iid-strong")

federated training command:
```bash
python main.py --data-path <data_path> --rounds 10 --clients 3 --strategy sync --distribution iid
```

isolated training command:
```bash
python main.py --data-path <data_path> --rounds 10 --clients 3 --strategy async --distribution iid --isolated
```


### 4. Results

Experiment results and metrics will be saved in the `results/` directory.

## Notes

- Ignore files and folders as specified in `.gitignore` (e.g., data/processed/, .venv/, .ipynb_checkpoints/, etc.)
- For more details, see the code in `src/`.
