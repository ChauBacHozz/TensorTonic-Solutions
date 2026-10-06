import torch
import torch.nn as nn

def train_with_scheduler(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.StepLR, num_epochs: int) -> dict:
    """
    Returns losses and lrs as lists of Python floats in a dictionary.
    """
    losses = []
    lrs = []
    # init_lr = optimizer.param_groups["lr"]
    for e in range(num_epochs):
        model.train()
        mean_loss = 0
        for input, target in dataloader:
            y_pred = model(input)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            mean_loss += loss.item()
        losses.append(mean_loss/len(dataloader))
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    return {
        "losses": losses,
        "lrs": lrs
    }
    
