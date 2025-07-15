import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1).values
    y = df["Class"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def non_iid_split(X_train, y_train, skew_type="weak", n_clients=3, random_state=42):
    df_train = pd.DataFrame(X_train)
    df_train["Class"] = y_train

    if skew_type == "weak":
        # Balanced but shuffled class splits
        fraud = df_train[df_train["Class"] == 1]
        normal = df_train[df_train["Class"] == 0]

        fraud_split = np.array_split(fraud, n_clients)
        normal_split = np.array_split(normal, n_clients)

        clients = [
            pd.concat([fraud_split[i], normal_split[i]])
            for i in range(n_clients)
        ]

    elif skew_type == "medium":
        # First client gets most fraud, others get varying normal/fraud mix
        fraud = df_train[df_train["Class"] == 1].sample(frac=1, random_state=random_state)
        normal = df_train[df_train["Class"] == 0].sample(frac=1, random_state=random_state)

        split_size_fraud = int(0.5 * len(fraud))
        split_size_normal = int(0.5 * len(normal))

        client_1 = fraud[:split_size_fraud]
        client_2 = normal[:split_size_normal]
        remaining = df_train.drop(client_1.index).drop(client_2.index)
        remaining_split = np.array_split(remaining.sample(frac=1, random_state=random_state), n_clients - 2)

        clients = [client_1, client_2] + remaining_split

    elif skew_type == "strong":
        # One client gets all fraud, others only normal
        fraud = df_train[df_train["Class"] == 1].sample(frac=1, random_state=random_state)
        normal = df_train[df_train["Class"] == 0].sample(frac=1, random_state=random_state)

        client_1 = fraud
        normal_splits = np.array_split(normal, n_clients - 1)
        clients = [client_1] + normal_splits

    else:
        raise ValueError(f"Unknown skew_type: {skew_type}")

    # Convert each client's DataFrame to TensorDataset
    client_datasets = []
    for client_df in clients:
        client_df = client_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        X_client = client_df.drop(columns=["Class"]).values
        y_client = client_df["Class"].values
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
