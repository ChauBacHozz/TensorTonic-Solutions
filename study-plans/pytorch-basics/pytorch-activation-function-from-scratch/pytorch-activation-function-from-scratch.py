import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    match method:
        case "relu":
            return torch.where(x > 0, x, 0)
        case "sigmoid":
            return 1 / (1 + torch.exp(-x))
        case "tanh":
            magnitude = torch.where(x >= 0, x, -x)
            decay = torch.exp(-2 * magnitude)
            positive = (1- decay) / (1 + decay)
            return torch.where(x >= 0, positive, -positive)
        case "leaky_relu":
            return torch.where(x > 0, x, 0.01*x)
