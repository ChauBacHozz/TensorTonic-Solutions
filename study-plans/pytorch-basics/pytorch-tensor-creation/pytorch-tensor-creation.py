import torch

def create_tensor(method: str, shape: list, value: float = 0.0) -> torch.Tensor:
    dict = {"zeros" : lambda : torch.zeros(shape, dtype = torch.float32),
           "ones" : lambda : torch.ones(shape, dtype = torch.float32),
           "full" : lambda : torch.full((shape), value, dtype = torch.float32)}
    return dict[method]()
