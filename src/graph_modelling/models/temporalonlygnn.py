import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch_geometric.nn import ChebConv
from typing import Optional


class TemporalOnlyGRU(nn.Module):
    """
    Baseline model using only a GRU applied independently to each node.
    Ignores spatial graph structure.
    """

    def __init__(self, num_nodes, num_vars, lags, horizon, hidden_channels=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.horizon = horizon
        self.hidden_channels = hidden_channels

        # GRU processes each node's time series independently
        self.gru = nn.GRU(num_vars, hidden_channels, batch_first=True)
        self.output_layer = nn.Linear(hidden_channels, horizon)

    def forward(self, x, edge_index=None, edge_weight=None, lambda_max=None):
        """
        Args:
            x: Input (B, T, N*F) - T is lags
            edge_index, edge_weight, lambda_max: Ignored by this model
        Returns:
            Output (B, horizon, N)
        """
        B, T, NF = x.shape
        assert T == self.lags and NF == self.num_nodes * self.num_vars

        # Reshape input to treat each node independently:
        # (B, T, N, F) -> (B, N, T, F) -> (B*N, T, F)
        x = x.view(B, T, self.num_nodes, self.num_vars)
        x = x.permute(0, 2, 1, 3)  # (B, N, T, F)
        x_flat = x.reshape(B * self.num_nodes, T, self.num_vars)  # (B*N, T, F)

        # Apply GRU
        # output: (B*N, T, hidden), h_n: (1, B*N, hidden)
        _, h_n = self.gru(x_flat)

        # Use the final hidden state (remove the first dimension)
        h_final = h_n.squeeze(0)  # (B*N, hidden)

        # Apply output layer
        y_flat = self.output_layer(h_final)  # (B*N, horizon)

        # Reshape back to (B, N, horizon)
        y = y_flat.view(B, self.num_nodes, self.horizon)

        # Permute to (B, horizon, N)
        y = y.permute(0, 2, 1).contiguous()
        return y
