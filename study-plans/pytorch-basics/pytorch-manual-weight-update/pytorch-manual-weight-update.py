import torch
import torch.nn as nn

def manual_train_step(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module, lr: float) -> float:
    """
    Returns the pre-update batch loss as a Python float.
    """
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= lr*param.grad
            param.grad.zero_()
    return loss.item()
    
