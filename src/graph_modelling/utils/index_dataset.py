import torch
import numpy as np
from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(
        self, indices, data, horizon, lazy=False, gpu=False, lags=0, target_offset=0
    ):
        self.indices = indices
        self.data = data
        self.lags = lags
        self.horizon = horizon
        self.target_offset = 72 - 24 + 1
        self.lazy = lazy
        self.gpu = gpu

    def __len__(self):
        return self.indices.shape[0]

    # Inside IndexDataset.__getitem__
    def __getitem__(self, x):
        idx = self.indices[x]
        y_start = idx + self.target_offset
        y_end = y_start + self.horizon

        if self.gpu:
            input_data = self.data[idx : idx + self.lags, ...]  # Corrected X slice
            target_data = self.data[y_start:y_end, ...]
            return input_data, target_data
        else:
            # if utilizing DDP-batching...
            if self.lazy:
                input_data = self.data[
                    idx : idx + self.lags, ...
                ].compute()  # Corrected X slice
                target_data = self.data[y_start:y_end, ...].compute()
                return torch.from_numpy(input_data), torch.from_numpy(target_data)
            else:
                input_data = self.data[idx : idx + self.lags, ...]  # Corrected X slice
                target_data = self.data[y_start:y_end, ...]
                return torch.from_numpy(input_data), torch.from_numpy(target_data)
