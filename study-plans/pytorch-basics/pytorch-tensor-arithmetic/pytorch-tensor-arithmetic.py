import torch

def tensor_op(x: torch.Tensor, y: torch.Tensor, op: str) -> torch.Tensor:
    dict = {"add" : lambda : torch.add(x,y),
           "multiply" : lambda : torch.mul(x,y),
           "matmul" : lambda : torch.matmul(x,y),
           "power" : lambda : torch.pow(x,y),
           "max" : lambda : torch.max(x,y)}
    return dict[op]()
