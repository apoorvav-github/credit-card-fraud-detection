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

def split_data(X, y, n_clients=3, iid=True, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    if iid:
        X_parts = np.array_split(X_train, n_clients)
        y_parts = np.array_split(y_train, n_clients)
    else:
        idx_pos = np.where(y_train == 1)[0]
        idx_neg = np.where(y_train == 0)[0]
        np.random.shuffle(idx_pos)
        np.random.shuffle(idx_neg)
        parts = []
        for i in range(n_clients):
            pos_slice = idx_pos[i::n_clients]
            neg_slice = idx_neg[i::n_clients]
            parts.append(np.concatenate([pos_slice, neg_slice]))
        X_parts = [X_train[p] for p in parts]
        y_parts = [y_train[p] for p in parts]
    client_datasets = [
        TensorDataset(torch.Tensor(x), torch.Tensor(y))
        for x, y in zip(X_parts, y_parts)
    ]
    central_test = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    return client_datasets, central_test
