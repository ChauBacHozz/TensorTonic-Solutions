import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.layer_1 = nn.Linear(in_features,hidden_size)
        self.activation = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size,out_features)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_layer_1 = self.layer_1(x)
        activation_1 = self.activation(out_layer_1)
        out_layer_2 = self.layer_2(activation_1)
        return out_layer_2
