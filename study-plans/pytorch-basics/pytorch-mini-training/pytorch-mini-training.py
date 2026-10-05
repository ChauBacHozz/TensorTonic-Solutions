import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    mean_loss = 0

    model.train()

    for input, target in dataloader:
        optimizer.zero_grad()
        y_pred = model(input)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()

    return mean_loss / len(dataloader)