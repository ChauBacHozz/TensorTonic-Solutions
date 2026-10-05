import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    # Đưa model vào trạng thái train
    model.train()
    running_loss = 0

    # Loop qua dataloader:
    for input, target in dataloader:
        # Batch i
        optimizer.zero_grad()
        y_pred = model(input)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataloader)