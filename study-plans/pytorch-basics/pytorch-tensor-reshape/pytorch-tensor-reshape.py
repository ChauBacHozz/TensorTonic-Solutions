import torch

def reshape_tensor(x: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the reshaped float32 tensor, including a scalar tensor after a complete squeeze.
    """
    op_prep = op.lower()
    match op_prep:
        case "view":
            return x.view(x.T.Shape)
        case "flatten":
            return x.flatten()
        case "squeeze":
            return x.squeeze()
        case "unsqueeze":
            return x.unsqueeze()
        case "transpose":
            return x.T
            
            
