import torch

def compute_gradient(values: torch.Tensor) -> torch.Tensor:
    values.requires_grad_(True)
    func = torch.sum(values**3 + 2*values)
    func.backward()
    return values.grad
