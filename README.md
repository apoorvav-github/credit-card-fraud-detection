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
│   └── processed/
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
python src/main.py --data-path <path_to_your_data> --clients 3 --rounds 5 --strategy sync --iid
```

- `--clients`: Number of federated clients
- `--rounds`: Number of FL rounds
- `--strategy`: FL strategy (`sync`, `async`, `hybrid`)
- `--iid`: Use IID data split

### 4. Results

Experiment results and metrics will be saved in the `results/` directory.

## Notes

- Ignore files and folders as specified in `.gitignore` (e.g., data/processed/, .venv/, .ipynb_checkpoints/, etc.)
- For more details, see the code in `src/`.
