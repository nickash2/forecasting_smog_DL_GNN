import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch_geometric.nn import ChebConv
from typing import Optional


import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch_geometric.nn import ChebConv
from typing import Optional

class SpatialOnlyGCN(nn.Module):
    """
    Fixed version of SpatialOnlyGCN with explicit dtype handling
    """

    def __init__(self, num_nodes, num_vars, horizon, K=1, hidden_channels=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.horizon = horizon
        
        # Create layers with explicit float32 dtype
        self.gcn1 = ChebConv(int(num_vars), int(hidden_channels), K=int(K))
        self.gcn2 = ChebConv(int(hidden_channels), int(hidden_channels), K=int(K))
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
        assert NF == self.num_nodes * self.num_vars, f"Expected {self.num_nodes * self.num_vars} features, got {NF}"

        # Take only the last time step's features
        x_last = x[:, -1, :]  # (B, N*F)
        x_last = x_last.view(B, self.num_nodes, self.num_vars)  # (B, N, F)

        # Reshape for GCN: (B*N, F)
        x_flat = x_last.reshape(-1, self.num_vars)

        # Apply GCN layers with explicit float32 conversion
        h1_flat = F_func.relu(
            self.gcn1(x_flat.float(), edge_index, edge_weight, batch=None, lambda_max=lambda_max)
        )
        h2_flat = F_func.relu(
            self.gcn2(h1_flat.float(), edge_index, edge_weight, batch=None, lambda_max=lambda_max)
        )

        # Apply output layer
        y_flat = self.output_layer(h2_flat.float())

        # Reshape back to (B, N, horizon)
        y = y_flat.view(B, self.num_nodes, self.horizon)

        # Permute to (B, horizon, N)
        y = y.permute(0, 2, 1).contiguous()
        return y