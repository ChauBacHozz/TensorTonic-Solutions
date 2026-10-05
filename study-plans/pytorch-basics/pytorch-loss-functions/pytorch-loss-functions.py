import torch

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float.
    """
    match method:
        case "mse":
            return ((pred - target) ** 2).mean().item()
        case "huber":
            a = torch.abs(pred - target)
            return (torch.where(torch.abs(pred - target) <= delta, a**2/2, delta * (a - delta/2))).mean().item()
        case "cross_entropy":
            pred = pred.to(torch.float64)
            batch_indices = torch.arange(pred.size(0))
            correct = pred[batch_indices, target]
            sum_log = (torch.logsumexp(pred, dim=1) - correct).mean().item()
            return sum_log
    