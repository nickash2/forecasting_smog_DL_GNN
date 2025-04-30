import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2 as PyGA3TGCN2


class A3TGCN(nn.Module):
    """Wrapper for PyTorch Geometric Temporal's A3TGCN2 implementation"""

    def __init__(
        self,
        num_nodes=3,
        num_vars=7,
        timesteps=72,
        horizon=24,
        gcn_out_channels=16,
        improved=False,
        cached=False,
        add_self_loops=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.horizon = horizon
        self.timesteps = timesteps

        # PyG A3TGCN2 expects different parameter format
        self.model = PyGA3TGCN2(
            in_channels=num_vars,
            out_channels=gcn_out_channels,
            periods=timesteps,
            batch_size=None,  # Will be determined during forward pass
            improved=improved,
            cached=cached,
            add_self_loops=add_self_loops,
        )

        # Additional linear layer to map from gcn_out_channels to horizon
        self.output_layer = nn.Linear(gcn_out_channels, horizon)

    def forward(self, x, edge_index=None, edge_weight=None, lambda_=None):
        """
        Args:
            x: Input data of shape (B, T, N*F) where B is batch size, T is timesteps,
               N is number of nodes, F is number of features per node
            edge_index: Graph edge indices
            edge_weight: Edge weights

        Returns:
            Tensor of shape (B, horizon, N)
        """
        B, T, NF = x.shape
        assert NF == self.num_nodes * self.num_vars, (
            f"Expected {self.num_nodes * self.num_vars}, got {NF}"
        )
        assert T == self.timesteps, f"Expected {self.timesteps} timesteps, got {T}"

        # Reshape to A3TGCN expected format: (B, N, F, T)
        x = x.reshape(B, T, self.num_nodes, self.num_vars)
        x = x.permute(0, 2, 3, 1)  # (B, N, F, T)

        # Update batch size in model if needed
        if hasattr(self.model, "batch_size") and self.model.batch_size != B:
            self.model.batch_size = B

        # Forward pass through A3TGCN2
        # Output will be (B, N, gcn_out_channels)
        h = self.model(x, edge_index, edge_weight)

        # Apply output layer to map to horizon
        # Reshape to (B*N, gcn_out_channels) for linear layer
        h_flat = h.reshape(-1, h.size(-1))
        out_flat = self.output_layer(h_flat)  # (B*N, horizon)

        # Reshape back to (B, N, horizon)
        out = out_flat.reshape(B, self.num_nodes, self.horizon)

        # Return with shape (B, horizon, N)
        return out.transpose(1, 2)
