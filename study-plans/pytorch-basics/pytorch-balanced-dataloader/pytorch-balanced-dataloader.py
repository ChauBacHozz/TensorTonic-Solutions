import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    """
    Returns a DataLoader with inverse-frequency sampling and replacement.
    """
    counts = torch.bincount(labels)
    w_s = 1 / counts

    sample_ws = [w_s[target] for target in labels]
    sampler = WeightedRandomSampler(
        weights=sample_ws, num_samples=len(sample_ws), replacement=True
    )

    return DataLoader(
        dataset=TensorDataset(features, labels), 
        batch_size=batch_size,
        sampler=sampler
    )
