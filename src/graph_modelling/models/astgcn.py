import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import ASTGCN as PyGASTGCN
from torch_geometric.utils import to_dense_adj


class ASTGCN(nn.Module):
    """Wrapper for PyTorch Geometric Temporal's ASTGCN implementation"""

    def __init__(
        self,
        num_nodes,
        num_vars,
        timesteps,
        K,
        num_blocks,
        horizon,
        block_channels,
        daily_span=24,
        weekly_span=24 * 7,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.horizon = horizon

        # PyG ASTGCN expects different parameter format
        self.model = PyGASTGCN(
            nb_block=num_blocks,
            in_channels=num_vars,
            K=K,
            nb_chev_filter=block_channels,
            nb_time_filter=block_channels,
            time_strides=1,
            num_for_predict=horizon,
            len_input=timesteps,
            num_of_vertices=num_nodes,
            normalization="sym",
        )

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

        # Reshape to PyG ASTGCN expected format: (B, N, F, T)
        x = x.reshape(B, T, self.num_nodes, self.num_vars)
        x = x.permute(0, 2, 3, 1)  # (B, N, F, T)

        # Forward pass through PyG ASTGCN
        out = self.model(x, edge_index)  # Output: (B, N, horizon)

        # Return with shape (B, horizon, N)
        return out.transpose(1, 2)
