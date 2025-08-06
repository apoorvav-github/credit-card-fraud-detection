import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def load_and_preprocess(path):
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1).values
    y = df["Class"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# def non_iid_split(X_train, y_train, skew_type="weak", n_clients=3, random_state=42):
#     df_train = pd.DataFrame(X_train)
#     df_train["Class"] = y_train

#     if skew_type == "weak":
#         # Balanced but shuffled class splits
#         fraud = df_train[df_train["Class"] == 1]
#         normal = df_train[df_train["Class"] == 0]

#         fraud_split = np.array_split(fraud, n_clients)
#         normal_split = np.array_split(normal, n_clients)

#         clients = [
#             pd.concat([fraud_split[i], normal_split[i]])
#             for i in range(n_clients)
#         ]

#     elif skew_type == "medium":
#         # First client gets most fraud, others get varying normal/fraud mix
#         fraud = df_train[df_train["Class"] == 1].sample(frac=1, random_state=random_state)
#         normal = df_train[df_train["Class"] == 0].sample(frac=1, random_state=random_state)

#         split_size_fraud = int(0.5 * len(fraud))
#         split_size_normal = int(0.5 * len(normal))

#         client_1 = fraud[:split_size_fraud]
#         client_2 = normal[:split_size_normal]
#         remaining = df_train.drop(client_1.index).drop(client_2.index)
#         remaining_split = np.array_split(remaining.sample(frac=1, random_state=random_state), n_clients - 2)

#         clients = [client_1, client_2] + remaining_split

#     elif skew_type == "strong":
#         # One client gets all fraud, others only normal
#         fraud = df_train[df_train["Class"] == 1].sample(frac=1, random_state=random_state)
#         normal = df_train[df_train["Class"] == 0].sample(frac=1, random_state=random_state)

#         client_1 = fraud
#         normal_splits = np.array_split(normal, n_clients - 1)
#         clients = [client_1] + normal_splits

#     else:
#         raise ValueError(f"Unknown skew_type: {skew_type}")

#     # Convert each client's DataFrame to TensorDataset
#     client_datasets = []
#     for client_df in clients:
#         client_df = client_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
#         X_client = client_df.drop(columns=["Class"]).values
#         y_client = client_df["Class"].values
#         dataset = TensorDataset(
#             torch.tensor(X_client, dtype=torch.float32),
#             torch.tensor(y_client, dtype=torch.float32)
#         )
#         client_datasets.append(dataset)

#     return client_datasets


def non_iid_split(X_train, y_train, skew_type="weak", n_clients=3, random_state=42):
    df_train = pd.DataFrame(X_train)
    df_train["Class"] = y_train
    df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True)  # Shuffle whole data

    total_samples = len(df_train)
    print(f"Total samples in training data: {total_samples}")
    clients = []

    if skew_type == "weak":
        # Stratified split: each client gets roughly equal samples preserving overall class ratio
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=random_state)
        for _, idx in skf.split(df_train.drop(columns=["Class"]), df_train["Class"]):
            client_df = df_train.iloc[idx].reset_index(drop=True)
            clients.append(client_df)

    elif skew_type == "medium":
        # Each client gets some fraud + normal but different fraud ratios between 5% and 50%
        np.random.seed(random_state)
        fraud_ratios = np.random.uniform(low=0.05, high=0.5, size=n_clients)

        # Split entire dataset indices into n_clients disjoint chunks
        indices = np.array_split(np.arange(total_samples), n_clients)

        for i, idx in enumerate(indices):
            client_df_candidate = df_train.iloc[idx].copy()

            client_size = len(client_df_candidate)
            desired_fraud_count = int(fraud_ratios[i] * client_size)
            desired_normal_count = client_size - desired_fraud_count

            fraud_samples = client_df_candidate[client_df_candidate["Class"] == 1]
            normal_samples = client_df_candidate[client_df_candidate["Class"] == 0]

            # Sample fraud and normal to meet desired ratio but capped by available samples
            fraud_selected = fraud_samples.sample(n=min(len(fraud_samples), desired_fraud_count), random_state=random_state)
            normal_selected = normal_samples.sample(n=min(len(normal_samples), desired_normal_count), random_state=random_state)

            client_df = pd.concat([fraud_selected, normal_selected]).sample(frac=1, random_state=random_state).reset_index(drop=True)

            # If undersized, fill remainder from leftover samples of the class if needed (optional, can skip)
            # For now just accept size < client_size if not enough samples.

            clients.append(client_df)

    elif skew_type == "strong":
        fraud = df_train[df_train["Class"] == 1].copy()
        normal = df_train[df_train["Class"] == 0].copy()

        # Shuffle fraud and normal samples
        fraud = fraud.sample(frac=1, random_state=random_state).reset_index(drop=True)
        normal = normal.sample(frac=1, random_state=random_state).reset_index(drop=True)

        clients = []

        # Number of minority samples per non-primary client
        minority_per_client = 5
        n_minority_clients = n_clients - 1

        # Assign most fraud to client 0
        fraud_for_client0 = fraud.iloc[: len(fraud) - (minority_per_client * n_minority_clients)]
        leftover_fraud = fraud.iloc[len(fraud) - (minority_per_client * n_minority_clients):]

        clients.append(fraud_for_client0.reset_index(drop=True))

        # Split normal samples to remaining clients
        normal_splits = np.array_split(normal, n_minority_clients)

        # Distribute leftover fraud evenly to each non-primary client
        for i in range(n_minority_clients):
            fraud_samples = leftover_fraud.iloc[i * minority_per_client: (i + 1) * minority_per_client]
            combined_df = pd.concat([
                normal_splits[i],
                fraud_samples
            ]).sample(frac=1, random_state=random_state).reset_index(drop=True)
            clients.append(combined_df)

        # Debug info
        total_samples = len(df_train)
        total_count = 0
        for i, client_df in enumerate(clients):
            counts = client_df["Class"].value_counts().to_dict()
            n_samples = len(client_df)
            total_count += n_samples
            print(f"Client {i} label distribution: {counts}, total samples: {n_samples}")

        print(f"Sum of all clients' samples: {total_count}")
        assert total_count == total_samples, "Sum of clients samples must equal original total"

    else:
        raise ValueError(f"Unknown skew_type: {skew_type}")

    # Convert to TensorDatasets with debug info
    client_datasets = []
    for i, client_df in enumerate(clients):
        client_df = client_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        X_client = client_df.drop(columns=["Class"]).values
        y_client = client_df["Class"].values

        unique, counts = np.unique(y_client, return_counts=True)
        print(f"Client {i} label distribution: {dict(zip(unique, counts))}, total samples: {len(client_df)}")

        dataset = TensorDataset(
            torch.tensor(X_client, dtype=torch.float32),
            torch.tensor(y_client, dtype=torch.float32)
        )
        client_datasets.append(dataset)

    return client_datasets

def split_data(X, y, n_clients=3, distribution="iid", random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    client_datasets = []
    if distribution == "iid":
        X_parts = np.array_split(X_train, n_clients)
        y_parts = np.array_split(y_train, n_clients)

        client_datasets = [
            TensorDataset(torch.tensor(x, dtype=torch.float32),torch.tensor(y, dtype=torch.float32))
            for x, y in zip(X_parts, y_parts)
            ]
    else:
        skew_type = distribution.replace("non-iid-", "")
        client_datasets = non_iid_split(X_train, y_train, skew_type, n_clients=n_clients)
    
    central_test = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    return client_datasets, central_test
