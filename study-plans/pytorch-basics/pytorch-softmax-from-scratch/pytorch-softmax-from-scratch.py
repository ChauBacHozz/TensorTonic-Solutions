import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    max, _ = torch.max(logits, dim=-1, keepdim=True)
    sum_exp = torch.sum(torch.exp(logits - max), dim=-1, keepdim=True)
    return torch.exp(logits - max) / sum_exp
    
