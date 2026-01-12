import torch
from torch.utils.data import Dataset


class SignDataset(Dataset):
    """
    PyTorch Dataset for sign language sequences
    X shape: (N, 30, 63)
    y shape: (N,)
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
