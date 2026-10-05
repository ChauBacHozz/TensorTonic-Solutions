import torch

def reshape_tensor(x: torch.Tensor, op: str) -> torch.Tensor:
    dict = {"transpose" : "T"}
    op = dict.get(op,op)
    attr = getattr(x,op)
    return attr() if callable(attr) else attr
