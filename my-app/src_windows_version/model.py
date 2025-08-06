import torch
import torch.nn as nn
import flwr as fl

class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)

def get_parameters(model: nn.Module) -> fl.common.Parameters:
    # Extract weights as numpy arrays
    params = [val.cpu().numpy() for val in model.state_dict().values()]
    # Wrap as Flower Parameters object
    return fl.common.ndarrays_to_parameters(params)

def set_parameters(model: nn.Module, parameters: fl.common.Parameters) -> None:
    # Convert Flower Parameters to numpy arrays
    params = fl.common.parameters_to_ndarrays(parameters)
    # Map back to state dict keys and convert to torch tensors
    state_dict = model.state_dict()
    new_state_dict = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), params)}
    model.load_state_dict(new_state_dict, strict=True)
