import torch
import numpy as np
from torch.utils.data import Dataset

class IndexDataset(Dataset):
    def __init__(self, indices, data, horizon, lazy=False, gpu=False, lags=0):
        self.indices = indices 
        self.data = data
        self.lags = lags
        self.horizon = horizon
        self.lazy = lazy
        self.gpu = gpu

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, x):
        idx = self.indices[x]
        x_start = idx
        x_end = idx + self.lags
        y_start = x_end
        y_end = y_start + self.horizon

        if self.gpu:
            return self.data[x_start:x_end,...], self.data[y_start:y_end,...]
        else:
            if self.lazy:
                return torch.from_numpy(self.data[x_start:x_end,...].compute()), torch.from_numpy(self.data[y_start:y_end,...].compute())
            else:
                return torch.from_numpy(self.data[x_start:x_end,...]), torch.from_numpy(self.data[y_start:y_end,...])
