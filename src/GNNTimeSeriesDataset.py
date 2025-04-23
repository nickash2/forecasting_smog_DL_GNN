import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from typing import List, Tuple, Optional
import pickle


class GNNTimeSeriesDataset(Dataset):
    """
    Dataset class for Graph Neural Networks with time series data.
    Creates sliding windows with customizable step size to prevent data leakage
    and reduce redundancy between samples.
    """

    def __init__(
        self,
        x: torch.Tensor,  # Input features with shape [timesteps, nodes, features]
        y: torch.Tensor,  # Target values with shape [timesteps, nodes, target_features]
        window_size: int,  # Length of input window
        forecast_horizon: int,  # Length of forecast horizon
        step: int = 24,  # Step size between windows (reduce redundancy)
        edge_index: torch.Tensor = None,  # Graph structure
        edge_attr: torch.Tensor = None,  # Edge features
    ):
        super(GNNTimeSeriesDataset, self).__init__()

        self.x = x
        self.y = y
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.step = step
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        # Create data samples with sliding windows
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        max_idx = len(self.x) - self.window_size - self.forecast_horizon + 1

        if max_idx <= 0:
            raise ValueError(
                f"Not enough data points ({len(self.x)}) for window size {self.window_size} and forecast horizon {self.forecast_horizon}"
            )

        for i in range(0, max_idx, self.step):
            # Get input window and forecast targets
            x_window = self.x[i : i + self.window_size]
            y_window = self.y[
                i + self.window_size : i + self.window_size + self.forecast_horizon
            ]

            # Permute x to get [nodes, timesteps, features] for GNN
            x_seq = x_window.permute(1, 0, 2).contiguous()

            # Create PyG Data object
            data = Data(
                x_seq=x_seq,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=y_window,
            )

            samples.append(data)

        print(f"Created {len(samples)} samples with step size {self.step}")
        return samples

    def len(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def get(self, idx: int) -> Data:
        """Return the idx-th sample."""
        return self.samples[idx]

    def normalize(self, x_min=None, x_max=None, y_min=None, y_max=None):
        """
        Normalize dataset using feature-wise normalization.
        If min/max not provided, compute them from this dataset.
        """
        if x_min is None or x_max is None:
            # Stack all x_seq tensors to find min/max
            all_x = torch.cat(
                [data.x_seq.view(-1, data.x_seq.size(-1)) for data in self.samples],
                dim=0,
            )
            x_min = all_x.min(dim=0, keepdim=True)[0].numpy()
            x_max = all_x.max(dim=0, keepdim=True)[0].numpy()

        if y_min is None or y_max is None:
            # Stack all y tensors to find min/max
            all_y = torch.cat(
                [data.y.view(-1, data.y.size(-1)) for data in self.samples], dim=0
            )
            y_min = all_y.min(dim=0, keepdim=True)[0].numpy()
            y_max = all_y.max(dim=0, keepdim=True)[0].numpy()

        # Apply normalization to each sample
        for data in self.samples:
            # Normalize x_seq
            x_arr = data.x_seq.numpy()
            x_norm = (x_arr - x_min) / (x_max - x_min + 1e-8)
            data.x_seq = torch.tensor(x_norm, dtype=torch.float)

            # Create flattened version for GCN
            data.x = torch.tensor(x_norm, dtype=torch.float).view(x_arr.shape[0], -1)

            # Normalize y
            y_arr = data.y.numpy()
            y_norm = (y_arr - y_min) / (y_max - y_min + 1e-8)
            data.y = torch.tensor(y_norm, dtype=torch.float)
            data.y_orig = torch.tensor(
                y_arr, dtype=torch.float
            )  # Keep original for evaluation

        return self, (x_min, x_max), (y_min, y_max)

    @classmethod
    def from_temporal_data(
        cls,
        train_data_path: str,
        val_data_path: str = None,
        test_data_path: str = None,
        chronological_split: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        **kwargs,
    ):
        """
        Create train/val/test datasets from raw temporal data with proper chronological splitting.

        Args:
            train_data_path: Path to raw data or pre-split training data
            val_data_path: Path to pre-split validation data (optional)
            test_data_path: Path to pre-split test data (optional)
            chronological_split: Whether to split data chronologically (True) or use provided splits
            train_ratio: Ratio of data to use for training if splitting chronologically
            val_ratio: Ratio of data to use for validation if splitting chronologically
            **kwargs: Additional arguments for the GNNTimeSeriesDataset constructor
        """
        # Implementation would go here...
        pass

    @staticmethod
    def save_datasets(
        train_dataset, val_dataset, test_dataset, save_dir, y_min=None, y_max=None
    ):
        """Save processed datasets and normalization parameters."""
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "train_dataset.pkl", "wb") as f:
            pickle.dump(train_dataset, f)

        with open(save_dir / "val_dataset.pkl", "wb") as f:
            pickle.dump(val_dataset, f)

        with open(save_dir / "test_dataset.pkl", "wb") as f:
            pickle.dump(test_dataset, f)

        if y_min is not None and y_max is not None:
            with open(save_dir / "y_min_max.pkl", "wb") as f:
                pickle.dump((y_min, y_max), f)

        print(f"Datasets saved to {save_dir}")
