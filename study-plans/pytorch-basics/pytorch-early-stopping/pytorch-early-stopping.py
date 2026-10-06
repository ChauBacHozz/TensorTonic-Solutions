import torch
import torch.nn as nn

def train_with_early_stopping(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    train_loss = []
    best_loss = float("inf")
    val_loss = []

    counter = 0
    stopped_epoch = 0
    
    for e in range(1, max_epochs + 1, 1):
        stopped_epoch = e
        model.train()
        train_loss_per_e = 0
        for train_x, train_y in train_loader:
            y_pred = model(train_x)
            loss = criterion(y_pred, train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_per_e+=loss.item() 
        train_loss_per_e /= len(train_loader)
        train_loss.append(train_loss_per_e)

        model.eval()
        with torch.no_grad():
            val_loss_mean = 0
            for val_x, val_y in val_loader:
                y_p = model(val_x)
                loss = criterion(y_p, val_y)
                val_loss_mean += loss.item()
            val_loss_mean /= len(val_loader)
            val_loss.append(val_loss_mean)

        if val_loss[-1] < best_loss:
            best_loss = val_loss[-1]
            counter = 0
        else:
            counter+=1

        if counter == patience:
            break
    return {
        "train_losses": train_loss,
        "val_losses": val_loss,
        "stopped_epoch": stopped_epoch
    }