import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    X_norm = (X - torch.mean(X, dim = 0))/torch.sqrt(torch.var(X, dim = 0, unbiased=False) + eps)
    return X_norm * gamma + beta