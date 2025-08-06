from collections import OrderedDict
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.exceptions import UndefinedMetricWarning
import warnings

import flwr as fl
from flwr.common import Parameters
from flwr.client import Client
from flwr.common import (
    GetPropertiesIns, GetPropertiesRes,
    GetParametersIns, GetParametersRes,
    FitIns, FitRes,
    EvaluateIns, EvaluateRes,
    Status, Code,
)

from model import FraudDetectionModel
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

def get_parameters(net: torch.nn.Module) -> fl.common.Parameters:
    # Return PyTorch model parameters as Flower Parameters object
    return ndarrays_to_parameters([val.cpu().numpy() for val in net.state_dict().values()])

def set_parameters(net: torch.nn.Module, parameters: fl.common.Parameters) -> None:
    # Set PyTorch model parameters from Flower Parameters object
    params_ndarrays = parameters_to_ndarrays(parameters)
    params_dict = zip(net.state_dict().keys(), params_ndarrays)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)

# ----------------------- Training / Evaluation -----------------------

def train(net: nn.Module, loader: DataLoader, epochs: int, device: torch.device):
    net.to(device)
    net.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device).float()
            optimizer.zero_grad()
            # preds = net(X).squeeze()
            preds = net(X).view(-1) 
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

# def evaluate_model(net: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
#     net.to(device)
#     net.eval()
#     ys, ps = [], []

#     with torch.no_grad():
#         for X, y in loader:
#             X = X.to(device)
#             p = net(X).squeeze().cpu().numpy()
#             ps.extend(p)
#             ys.extend(y.numpy())

#     auc = roc_auc_score(ys, ps)
#     acc = accuracy_score(ys, (np.array(ps) > 0.5).astype(int))
#     loss = 1.0 - auc  # Example loss (could use BCE or other if preferred)
#     metrics = {"auc": auc, "accuracy": acc}
#     return loss, metrics

# This I have redefined to include precision, recall, and F1 score: akash
def evaluate_model(net: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    net.to(device)
    net.eval()
    ys, ps = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            p = net(X).squeeze().cpu().numpy()
            ps.extend(p)
            ys.extend(y.numpy())

    ys = np.array(ys)
    ps = np.array(ps)
    preds = (ps > 0.7).astype(int)

    auc = roc_auc_score(ys, ps)
    precision = precision_score(ys, preds, average="macro", zero_division=0)
    recall = recall_score(ys, preds, average="macro", zero_division=0)
    f1 = f1_score(ys, preds, average="macro", zero_division=0)

    loss = 1.0 - auc  
    acc = accuracy_score(ys, (np.array(ps) > 0.7).astype(int))
    metrics = {
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": acc
    }

    return loss, metrics

# ----------------------- Flower Client -----------------------

class FLClient(Client):
    def __init__(self, cid: str, train_loader: DataLoader, test_loader: DataLoader, device=torch.device("cpu")):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model = FraudDetectionModel()

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        num_samples = len(self.train_loader.dataset)
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={"num_samples": str(num_samples)},
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        params = get_parameters(self.model)
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=params,
        )

    def fit(self, ins: FitIns) -> FitRes:
        # ins.parameters is a Parameters object
        set_parameters(self.model, ins.parameters)
        local_epochs = int(ins.config.get("local_epochs", 1))
        train(self.model, self.train_loader, local_epochs, self.device)
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=get_parameters(self.model),
            num_examples=len(self.train_loader.dataset),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        set_parameters(self.model, ins.parameters)
        loss, metrics = evaluate_model(self.model, self.test_loader, self.device)
        metrics["cid"] = self.cid
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=loss,
            num_examples=len(self.test_loader.dataset),
            metrics=metrics,
        )
