import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch_geometric.nn import ChebConv, GCNConv, GATConv
from typing import Optional


class TemporalAttention(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.attn = nn.Linear(hidden_channels, 1)  # Simple attention mechanism

    def forward(self, x):
        # x: [B, T, N, hidden_channels]
        B, T, N, F = x.shape
        x_flat = x.view(B * N, T, F)  # [B*N, T, F]

        # Compute attention scores for each time step
        attn_weights = self.attn(x_flat)  # [B*N, T, 1]
        attn_weights = F_func.softmax(
            attn_weights, dim=1
        )  # Softmax over time dimension

        # Apply attention weights to the features
        weighted_x = torch.bmm(attn_weights.permute(0, 2, 1), x_flat)  # [B*N, 1, F]
        weighted_x = weighted_x.squeeze(1)  # [B*N, F]

        return weighted_x


class SimpleChebGRUWithTemporalAttention(nn.Module):
    def __init__(self, num_nodes, num_vars, lags, horizon, hidden_channels=16, K=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.horizon = horizon
        self.hidden_channels = hidden_channels

        # Single ChebConv layer for spatial processing
        self.chebconv = GCNConv(in_channels=num_vars, out_channels=hidden_channels)

        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(hidden_channels)

        # GRU for temporal processing (node-wise)
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)

        # Output projection
        self.output_layer = nn.Linear(hidden_channels, horizon)

    def forward(self, x, edge_index, edge_weight=None):
        B, T, NF = x.shape
        assert NF == self.num_nodes * self.num_vars

        # Reshape input to [B, T, N, F]
        x = x.view(B, T, self.num_nodes, self.num_vars)

        cheb_outputs = []

        for t in range(T):
            xt = x[:, t, :, :]  # [B, N, F]
            xt = xt.reshape(B * self.num_nodes, self.num_vars)

            # Apply spatial ChebConv
            out = self.chebconv(xt, edge_index, edge_weight)  # [B*N, hidden]
            out = F_func.relu(out).view(B, self.num_nodes, self.hidden_channels)
            cheb_outputs.append(out.unsqueeze(1))  # [B, 1, N, hidden]

        # Stack over time: [B, T, N, hidden]
        cheb_seq = torch.cat(cheb_outputs, dim=1)

        # Apply Temporal Attention (focus on important time steps)
        x_attention = self.temporal_attention(cheb_seq)

        # Reshape for GRU: [B*N, T, hidden]
        x_attention = x_attention.view(B * self.num_nodes, -1)

        # GRU over time
        _, h_n = self.gru(x_attention.unsqueeze(1))  # GRU expects [B*N, T, hidden]
        h_final = h_n.squeeze(0)  # [B*N, hidden]

        # Output: [B*N, horizon] → reshape → [B, N, horizon]
        y = self.output_layer(h_final).view(B, self.num_nodes, self.horizon)

        return y.permute(0, 2, 1)  # [B, horizon, N]


class GATGRUModel(nn.Module):
    def __init__(self, num_nodes, num_vars, lags, horizon, hidden_channels=16, heads=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.horizon = horizon
        self.hidden_channels = hidden_channels

        # Replace ChebConv with GATConv (built-in attention mechanism)
        self.gatconv = GATConv(
            in_channels=num_vars,
            out_channels=hidden_channels
            // heads,  # Divide by heads to keep same param count
            heads=heads,
            concat=True,
        )

        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(hidden_channels)

        # GRU for temporal processing
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)

        # Output projection
        self.output_layer = nn.Linear(hidden_channels, horizon)

    def forward(self, x, edge_index, edge_weight=None):
        B, T, NF = x.shape
        assert NF == self.num_nodes * self.num_vars

        # Reshape input to [B, T, N, F]
        x = x.view(B, T, self.num_nodes, self.num_vars)

        gat_outputs = []

        for t in range(T):
            xt = x[:, t, :, :]  # [B, N, F]
            xt = xt.reshape(B * self.num_nodes, self.num_vars)

            # Apply GATConv (has built-in attention mechanism)
            # Note: edge_weight is typically not used in GATConv as it learns attention weights
            out = self.gatconv(xt, edge_index)  # [B*N, hidden]
            out = F_func.relu(out).view(B, self.num_nodes, self.hidden_channels)
            gat_outputs.append(out.unsqueeze(1))  # [B, 1, N, hidden]

        # Stack over time: [B, T, N, hidden]
        gat_seq = torch.cat(gat_outputs, dim=1)

        # Apply Temporal Attention (focus on important time steps)
        temporal_features = self.temporal_attention(gat_seq)

        # Reshape for GRU: [B*N, hidden]
        features = temporal_features.view(B * self.num_nodes, -1)

        # GRU over time
        _, h_n = self.gru(features.unsqueeze(1))  # GRU expects [B*N, T, hidden]
        h_final = h_n.squeeze(0)  # [B*N, hidden]

        # Output: [B*N, horizon] → reshape → [B, N, horizon]
        y = self.output_layer(h_final).view(B, self.num_nodes, self.horizon)

        return y.permute(0, 2, 1)  # [B, horizon, N]
