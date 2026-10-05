import torch

def tensor_op(x: torch.Tensor, y: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the operation result as a float32 tensor.
    """
    op_prep = op.lower()
    match op_prep:
        case "add":
            return torch.add(x, y)
        case "multiply":
            return torch.mul(x, y)
        case "matmul":
            return torch.matmul(x, y)
        case "power":
            return torch.pow(x, y)
        case "max":
            return torch.maximum(x, y)
