import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, data: list, label_col: int):
        self.data_full = torch.Tensor(data)
        self.label = self.data_full[:, label_col].reshape(self.data_full.size(0),1)
        columns = list(range(self.data_full.size(1)))
        columns.remove(label_col)
        self.features = self.data_full[:, columns]
        self.n_samples = len(self.data_full)
    def __len__(self) -> int:
        """
        Returns the number of rows.
        """
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (features, label) as float32 tensors of shapes (D,) and (1,).
        """
        return self.features[idx, :], self.label[idx]
