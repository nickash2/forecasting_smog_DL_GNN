import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch_geometric.nn import ChebConv
from typing import Optional


class SpatialOnlyGCN(nn.Module):
    """
    Baseline model using only graph convolutions on the last time step.
    Ignores temporal history.
    """

    def __init__(self, num_nodes, num_vars, horizon, K=2, hidden_channels=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.horizon = horizon

        self.gcn1 = ChebConv(num_vars, hidden_channels, K=K)
        self.gcn2 = ChebConv(hidden_channels, hidden_channels, K=K)
        self.output_layer = nn.Linear(hidden_channels, horizon)

    def forward(self, x, edge_index, edge_weight=None, lambda_max=None):
        """
        Args:
            x: Input (B, T, N*F) - T is lags
            edge_index, edge_weight, lambda_max: Graph info
        Returns:
            Output (B, horizon, N)
        """
        B, T, NF = x.shape
        assert NF == self.num_nodes * self.num_vars

        # Take only the last time step's features
        x_last = x[:, -1, :]  # (B, N*F)
        x_last = x_last.view(B, self.num_nodes, self.num_vars)  # (B, N, F)

        # Reshape for GCN: (B*N, F)
        x_flat = x_last.reshape(-1, self.num_vars)

        # Apply GCN layers
        h1_flat = F_func.relu(
            self.gcn1(
                x_flat, edge_index, edge_weight, batch=None, lambda_max=lambda_max
            )
        )
        h2_flat = F_func.relu(
            self.gcn2(
                h1_flat, edge_index, edge_weight, batch=None, lambda_max=lambda_max
            )
        )  # (B*N, hidden)

        # Apply output layer
        y_flat = self.output_layer(h2_flat)  # (B*N, horizon)

        # Reshape back to (B, N, horizon)
        y = y_flat.view(B, self.num_nodes, self.horizon)

        # Permute to (B, horizon, N)
        y = y.permute(0, 2, 1).contiguous()
        return y
