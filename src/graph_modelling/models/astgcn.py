import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch_geometric.nn import ChebConv
from typing import Optional


class SpatialAttentionLayer(nn.Module):
    """Computes spatial attention scores."""

    def __init__(self, num_nodes, in_channels, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(in_channels, hidden_dim)
        self.W_k = nn.Linear(in_channels, hidden_dim)
        self.W_v = nn.Linear(in_channels, in_channels)  # Output same dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, T, N, F)
        Returns:
            Attention output (B, T, N, F)
        """
        B, T, N, F = x.shape

        # Calculate Q, K, V for each time step independently
        # Reshape to combine B and T for batch matrix multiplication
        x_reshaped = x.reshape(B * T, N, F)  # (B*T, N, F)

        query = self.W_q(x_reshaped)  # (B*T, N, H_dim)
        key = self.W_k(x_reshaped)  # (B*T, N, H_dim)
        value = self.W_v(x_reshaped)  # (B*T, N, F)

        # Calculate attention scores: Q * K^T / sqrt(d_k)
        # key.transpose(-2, -1) -> (B*T, H_dim, N)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_dim**0.5)
        # scores shape: (B*T, N, N)

        # Apply softmax
        attn_weights = F_func.softmax(scores, dim=-1)  # (B*T, N, N)

        # Calculate weighted sum: Attention * V
        attn_output = torch.matmul(attn_weights, value)  # (B*T, N, F)

        # Reshape back to original format
        attn_output = attn_output.view(B, T, N, F)

        return attn_output + x  # Add residual connection


class TemporalAttentionLayer(nn.Module):
    """Computes temporal attention scores."""

    def __init__(self, num_nodes, in_channels, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(in_channels, hidden_dim)
        self.W_k = nn.Linear(in_channels, hidden_dim)
        self.W_v = nn.Linear(in_channels, in_channels)  # Output same dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, T, N, F)
        Returns:
            Attention output (B, T, N, F)
        """
        B, T, N, F = x.shape

        # Permute and reshape for temporal attention: (B, N, T, F)
        x_permuted = x.permute(0, 2, 1, 3)  # (B, N, T, F)
        x_reshaped = x_permuted.reshape(B * N, T, F)  # (B*N, T, F)

        query = self.W_q(x_reshaped)  # (B*N, T, H_dim)
        key = self.W_k(x_reshaped)  # (B*N, T, H_dim)
        value = self.W_v(x_reshaped)  # (B*N, T, F)

        # Calculate attention scores: Q * K^T / sqrt(d_k)
        # key.transpose(-2, -1) -> (B*N, H_dim, T)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_dim**0.5)
        # scores shape: (B*N, T, T)

        # Apply softmax
        attn_weights = F_func.softmax(scores, dim=-1)  # (B*N, T, T)

        # Calculate weighted sum: Attention * V
        attn_output = torch.matmul(attn_weights, value)  # (B*N, T, F)

        # Reshape and permute back
        attn_output = attn_output.view(B, N, T, F)
        attn_output = attn_output.permute(0, 2, 1, 3)  # (B, T, N, F)

        return attn_output + x  # Add residual connection


class SpatioTemporalBlock(nn.Module):
    """Combines spatial attention, GCN, temporal attention, and TCN (GRU)."""

    def __init__(self, num_nodes, in_channels, hidden_channels, K, gru_hidden_channels):
        super().__init__()
        attn_hidden = hidden_channels // 2  # Example dimension for attention Q,K

        self.spatial_attn = SpatialAttentionLayer(num_nodes, in_channels, attn_hidden)
        self.graph_conv = ChebConv(in_channels, hidden_channels, K=K)
        self.temporal_attn = TemporalAttentionLayer(
            num_nodes, hidden_channels, attn_hidden
        )
        # GRU acts as the temporal convolution/aggregation layer
        self.temporal_gru = nn.GRU(
            hidden_channels, gru_hidden_channels, batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm([num_nodes, hidden_channels])
        self.layer_norm2 = nn.LayerNorm([num_nodes, gru_hidden_channels])
        self.gru_hidden_channels = gru_hidden_channels

    def forward(self, x, edge_index, edge_weight=None, lambda_max=None):
        """
        Args:
            x: Input (B, T, N, F_in)
            edge_index, edge_weight, lambda_max: Graph info
        Returns:
            Output (B, T, N, F_out) where F_out is gru_hidden_channels
        """
        B, T, N, F_in = x.shape

        # 1. Spatial Attention
        x_sp_attn = self.spatial_attn(x)  # (B, T, N, F_in)

        # 2. Graph Convolution (applied per time step)
        x_gcn_list = []
        for t in range(T):
            x_t = x_sp_attn[:, t, :, :]  # (B, N, F_in)
            x_t_flat = x_t.reshape(-1, F_in)  # (B*N, F_in)
            gcn_out_flat = self.graph_conv(
                x_t_flat, edge_index, edge_weight, batch=None, lambda_max=lambda_max
            )  # (B*N, H)
            gcn_out = gcn_out_flat.reshape(B, N, -1)  # (B, N, H)
            x_gcn_list.append(gcn_out)
        x_gcn = torch.stack(x_gcn_list, dim=1)  # (B, T, N, H)
        x_gcn = self.layer_norm1(x_gcn)  # Apply LayerNorm

        # 3. Temporal Attention
        x_temp_attn = self.temporal_attn(x_gcn)  # (B, T, N, H)

        # 4. Temporal Aggregation (GRU per node)
        # Reshape for GRU: (B*N, T, H)
        gru_input = x_temp_attn.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        gru_output, _ = self.temporal_gru(gru_input)  # gru_output: (B*N, T, gru_H)

        # Reshape back: (B, N, T, gru_H) -> (B, T, N, gru_H)
        output = gru_output.view(B, N, T, self.gru_hidden_channels).permute(0, 2, 1, 3)
        output = self.layer_norm2(output)

        return output


class ASTGCN_Like(nn.Module):
    """Simplified ASTGCN-like model."""

    def __init__(
        self,
        num_nodes,
        num_vars,
        lags,
        horizon,
        block_channels=32,
        gru_channels=32,
        K=2,
        num_blocks=1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.horizon = horizon

        # Input embedding (optional, could just use num_vars)
        self.input_embedding = nn.Linear(num_vars, block_channels)

        self.blocks = nn.ModuleList()
        current_channels = block_channels
        for _ in range(num_blocks):
            self.blocks.append(
                SpatioTemporalBlock(
                    num_nodes, current_channels, block_channels, K, gru_channels
                )
            )
            current_channels = gru_channels  # Output of GRU becomes input to next block

        # Final prediction layer(s)
        # Use last time step's output from the final block
        self.final_conv1 = nn.Conv2d(
            in_channels=gru_channels, out_channels=128, kernel_size=(1, 1)
        )
        self.final_conv2 = nn.Conv2d(
            in_channels=128, out_channels=horizon, kernel_size=(1, 1)
        )

    def forward(self, x, edge_index, edge_weight=None, lambda_max=None):
        """
        Args:
            x: Input (B, T, N*F) where T=lags
            edge_index, edge_weight, lambda_max: Graph info
        Returns:
            Output (B, horizon, N)
        """
        B, T, NF = x.shape
        assert T == self.lags and NF == self.num_nodes * self.num_vars

        # Reshape and embed input: (B, T, N, F) -> (B, T, N, block_channels)
        x = x.view(B, T, self.num_nodes, self.num_vars)
        x = self.input_embedding(x)

        # Pass through spatio-temporal blocks
        for block in self.blocks:
            x = block(x, edge_index, edge_weight, lambda_max)
        # x shape after blocks: (B, T, N, gru_channels)

        # Instead of only last step, keep ALL steps
        # Shape: (B, T, N, gru_channels)

        x = x.permute(0, 3, 1, 2)  # (B, gru_channels, T, N)

        x_out = F_func.relu(self.final_conv1(x))  # (B, 128, T, N)
        x_out = self.final_conv2(x_out)           # (B, horizon, T, N)

        # Now, pool over T dimension (aggregate over time!)
        x_out = x_out.mean(dim=2)  # (B, horizon, N)

        return x_out
