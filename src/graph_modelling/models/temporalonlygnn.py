import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv
from typing import Optional


class TemporalChebConvGRU(nn.Module):
    """
    A model that uses ChebConv for spatial feature extraction and GRU for temporal processing.
    It captures both spatial and temporal dependencies.
    """

    def __init__(
        self,
        num_nodes,
        num_vars,
        lags,
        horizon,
        hidden_channels=32,
        K=1,
        num_cheb_layers=2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.horizon = horizon
        self.hidden_channels = hidden_channels
        self.K = K
        self.num_cheb_layers = num_cheb_layers

        # Create ChebConv layers for spatial feature extraction
        self.cheb_convs = nn.ModuleList(
            [
                ChebConv(
                    in_channels=num_vars if i == 0 else hidden_channels,
                    out_channels=hidden_channels,
                    K=self.K,
                    normalization="sym",
                )
                for i in range(self.num_cheb_layers)
            ]
        )

        # GRU layer for temporal processing
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)

        # Output layer to predict the horizon
        self.output_layer = nn.Linear(hidden_channels, horizon)

    def forward(self, x, edge_index, edge_weight=None, lambda_max=None):
        """
        Forward pass through the TemporalChebConvGRU model.

        Args:
            x: Input (B, T, N*F) - B: batch size, T: time steps (lags), N: nodes, F: features (num_vars)
            edge_index: Graph connectivity information
            edge_weight: Edge weights (optional)
            lambda_max: Maximum eigenvalue of the Laplacian (optional)

        Returns:
            y: Predictions (B, horizon, N)
        """
        B, T, NF = x.shape
        assert T == self.lags and NF == self.num_nodes * self.num_vars

        # Reshape input to treat each node independently:
        x = x.view(B, T, self.num_nodes, self.num_vars)
        x = x.permute(0, 2, 1, 3)  # (B, N, T, F)

        # Apply ChebConv layers for spatial feature extraction
        for i, conv in enumerate(self.cheb_convs):
            # Process each time step through ChebConv
            out = []
            for t in range(T):
                # Get features at time t for all nodes: (B, N, F)
                features_t = x[:, :, t, :]

                # Reshape to (B*N, F) for processing by ChebConv
                features_t = features_t.reshape(B * self.num_nodes, -1)

                # Apply ChebConv - expects (num_nodes, in_channels)
                conv_out = conv(
                    features_t, edge_index, edge_weight, lambda_max=lambda_max
                )
                conv_out = F.relu(conv_out)  # (B*N, hidden_channels)

                # Reshape to (B, N, hidden_channels)
                conv_out = conv_out.view(B, self.num_nodes, self.hidden_channels)
                out.append(conv_out)

            # Stack along the time dimension: (B, N, T, hidden_channels)
            x = torch.stack(out, dim=2)

        # Flatten for GRU processing (B*N, T, hidden_channels)
        x_flat = x.contiguous().view(B * self.num_nodes, T, self.hidden_channels)

        # Apply GRU for temporal processing
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


class SimpleChebGRU(nn.Module):
    def __init__(self, num_nodes, num_vars, lags, horizon, hidden_channels=16, K=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.horizon = horizon
        self.hidden_channels = hidden_channels

        # Single ChebConv layer
        self.chebconv = GCNConv(in_channels=num_vars, out_channels=hidden_channels)

        # GRU for temporal processing (node-wise)
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)

        # Output projection
        self.output_layer = nn.Linear(hidden_channels, horizon)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Tensor of shape [B, T, N*F] = [batch, lags, nodes * features]
            edge_index: Graph edges (PyG format)
            edge_weight: Optional edge weights

        Returns:
            y: [B, horizon, N]
        """
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
            out = F.relu(out).view(B, self.num_nodes, self.hidden_channels)
            cheb_outputs.append(out.unsqueeze(1))  # [B, 1, N, hidden]

        # Stack over time: [B, T, N, hidden]
        cheb_seq = torch.cat(cheb_outputs, dim=1)

        # Reshape to [B*N, T, hidden]
        cheb_seq = cheb_seq.permute(0, 2, 1, 3).reshape(
            B * self.num_nodes, T, self.hidden_channels
        )

        # GRU over time
        _, h_n = self.gru(cheb_seq)  # h_n: [1, B*N, hidden]
        h_final = h_n.squeeze(0)  # [B*N, hidden]

        # Output: [B*N, horizon] → reshape → [B, N, horizon]
        y = self.output_layer(h_final).view(B, self.num_nodes, self.horizon)

        # [B, horizon, N]
        return y.permute(0, 2, 1)
