from collections import OrderedDict
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client
from flwr.common import (
    PropertiesIns, PropertiesRes,
    GetParametersIns, GetParametersRes,
    FitIns, FitRes,
    EvaluateIns, EvaluateRes,
)

from model_utils import MLP
from sklearn.metrics import roc_auc_score, accuracy_score

def get_parameters(net: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net: nn.Module, loader: DataLoader, epochs: int, device: torch.device):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.to(device).train()
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = net(X).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

def evaluate_model(net: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    net.to(device).eval()
    ys, ps = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            p = net(X).squeeze().cpu().numpy()
            ps.extend(p); ys.extend(y.numpy())
    auc = roc_auc_score(ys, ps)
    acc = accuracy_score(ys, np.array(ps) > 0.5)
    return 1.0 - auc, {"auc": auc, "accuracy": acc}

class FLClient(Client):
    def __init__(self, cid: str, train_loader: DataLoader, test_loader: DataLoader, device=torch.device("cpu")):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model = MLP()
        # initialize with random weights
        set_parameters(self.model, get_parameters(self.model))

    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        num_samples = len(self.train_loader.dataset)
        return PropertiesRes(properties={"num_samples": str(num_samples)})

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(parameters=get_parameters(self.model))

    def fit(self, ins: FitIns) -> FitRes:
        set_parameters(self.model, ins.parameters)
        local_epochs = int(ins.config.get("local_epochs", 1))
        train(self.model, self.train_loader, local_epochs, self.device)
        return FitRes(parameters=get_parameters(self.model),
                      num_examples=len(self.train_loader.dataset),
                      metrics={})

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        set_parameters(self.model, ins.parameters)
        loss, metrics = evaluate_model(self.model, self.test_loader, self.device)
        return EvaluateRes(loss=loss,
                           num_examples=len(self.test_loader.dataset),
                           metrics=metrics)
