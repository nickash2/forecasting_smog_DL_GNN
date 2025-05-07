import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch_geometric.nn import ChebConv, GCNConv
from typing import Optional, List


class SpatialOnlyGCN(nn.Module):
    """
    SpatialOnlyGCN with tunable number of ChebConv layers
    """

    def __init__(
        self, num_nodes, num_vars, horizon, K=1, hidden_channels=32, num_layers=2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.horizon = horizon
        self.num_layers = num_layers

        # Dynamic creation of ChebConv layers
        self.conv_layers = nn.ModuleList()

        # First layer: input features -> hidden
        self.conv_layers.append(ChebConv(int(num_vars), int(hidden_channels), K=int(K)))

        # Additional hidden layers
        for i in range(num_layers - 1):
            self.conv_layers.append(
                ChebConv(int(hidden_channels), int(hidden_channels), K=int(K))
            )

        # Output layer stays the same
        self.output_layer = nn.Linear(int(hidden_channels), int(horizon))

        # Force all parameters to float32
        for param in self.parameters():
            param.data = param.data.float()

    def forward(self, x, edge_index, edge_weight=None, lambda_max=None):
        # Ensure input is float32
        x = x.float()
        edge_index = edge_index.long()
        if edge_weight is not None:
            edge_weight = edge_weight.float()

        B, T, NF = x.shape
        assert NF == self.num_nodes * self.num_vars, (
            f"Expected {self.num_nodes * self.num_vars} features, got {NF}"
        )

        # Take only the last time step's features
        x_last = x[:, -1, :]  # (B, N*F)
        x_last = x_last.view(B, self.num_nodes, self.num_vars)  # (B, N, F)

        # Reshape for GCN: (B*N, F)
        h = x_last.reshape(-1, self.num_vars)

        # Apply all ChebConv layers
        for conv in self.conv_layers:
            h = F_func.relu(
                conv(
                    h.float(),
                    edge_index,
                    edge_weight,
                    batch=None,
                    lambda_max=lambda_max,
                )
            )

        # Apply output layer
        y_flat = self.output_layer(h.float())

        # Reshape back to (B, N, horizon)
        y = y_flat.view(B, self.num_nodes, self.horizon)

        # Permute to (B, horizon, N)
        y = y.permute(0, 2, 1).contiguous()
        return y
