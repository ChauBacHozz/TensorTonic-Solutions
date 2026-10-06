import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        torch.manual_seed(42)
        self.weight = nn.Parameter(torch.randn(out_features,in_features))
        self.bias = nn.Parameter(torch.randn(out_features,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(x,self.weight.T) + self.bias
        return output
