import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


class GraphTimeSeriesDataset(Dataset):
    def __init__(self, X, Y, window_size, forecast_horizon, step=1, edge_index=None):
        """
        Args:
            X (torch.Tensor): Tensor of shape (num_timesteps, num_nodes, num_features)
                              e.g., (num_timesteps, 3, num_features)
            Y (torch.Tensor): Tensor of shape (num_timesteps, num_nodes, target_features)
            window_size (int): Number of past timesteps to use as input.
            forecast_horizon (int): Number of future timesteps to predict.
            step (int): Step size between samples (to reduce redundancy).
            edge_index (torch.Tensor): Edge index tensor (shape [2, num_edges]).
                                       This should be provided (or computed elsewhere).
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.step = step
        self.edge_index = edge_index

        # Assume X and Y are already sorted chronologically.
        self.X = X  # Shape: (T, num_nodes, num_features)
        self.Y = Y  # Shape: (T, num_nodes, target_features)

        self.pairs = self._precompute_pairs()

    def _precompute_pairs(self):
        pairs = []
        num_timesteps = self.X.shape[0]
        # We can slide with the given step.
        for i in range(
            0, num_timesteps - self.window_size - self.forecast_horizon + 1, self.step
        ):
            # Select window for input and forecast horizon for target.
            X_window = self.X[i : i + self.window_size]
            Y_window = self.Y[
                i + self.window_size : i + self.window_size + self.forecast_horizon
            ]
            # X_window shape: (window_size, num_nodes, num_features)
            # Y_window shape: (forecast_horizon, num_nodes, target_features)

            # For each node, flatten the time window into a single vector.
            # That is, for each node: (window_size, num_features) -> (window_size * num_features,)
            # Similarly for the forecast horizon.
            num_nodes = X_window.shape[1]  # e.g., 3 (cities)
            X_flat = X_window.permute(1, 0, 2).reshape(num_nodes, -1)
            Y_flat = Y_window.permute(1, 0, 2).reshape(num_nodes, -1)

            # Create the Data object with the provided edge_index.
            data = Data(x=X_flat, y=Y_flat, edge_index=self.edge_index)
            pairs.append(data)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
