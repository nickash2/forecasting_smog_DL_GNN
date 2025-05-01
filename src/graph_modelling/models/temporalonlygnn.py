import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
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


class SpatiotemporalGCN(nn.Module):
    def __init__(self, num_nodes, num_vars, lags, horizon, hidden_channels=32, K=1):
        super(SpatiotemporalGCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.horizon = horizon
        self.hidden_channels = hidden_channels
        self.K = K

        # Spatial Convolution Layers (Chebyshev)
        self.chebconv = ChebConv(
            in_channels=num_vars,
            out_channels=hidden_channels,
            K=self.K,
            normalization="sym",
        )

        # Temporal convolution to model time steps for each node
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )

        # GRU for modeling combined spatiotemporal representations
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)

        # Output layer to predict the horizon
        self.output_layer = nn.Linear(hidden_channels, horizon)

    def forward(self, x, edge_index, edge_weight=None, lambda_max=None):
        B, T, NF = x.shape
        assert NF == self.num_nodes * self.num_vars
        x = x.view(B, T, self.num_nodes, self.num_vars)  # (B, T, N, F)

        # Apply spatial graph convolution for each time step (Chebyshev convolution)
        spatial_out = []
        for t in range(T):
            features_t = x[:, t, :, :]  # (B, N, F)
            features_t = features_t.reshape(B * self.num_nodes, -1)  # (B*N, F)
            spatial_out_t = self.chebconv(
                features_t, edge_index, edge_weight, lambda_max=lambda_max
            )
            # Add ReLU activation
            spatial_out_t = F.relu(spatial_out_t)
            spatial_out_t = spatial_out_t.view(B, self.num_nodes, self.hidden_channels)
            spatial_out.append(spatial_out_t)

        spatial_out = torch.stack(spatial_out, dim=1)  # (B, T, N, hidden_channels)

        # Process each node separately through temporal convolution
        temporal_out = []
        for n in range(self.num_nodes):
            # Extract data for this node: (B, T, hidden_channels)
            node_data = spatial_out[:, :, n, :]

            # Permute to (B, hidden_channels, T) for Conv1d
            node_data = node_data.permute(0, 2, 1)

            # Apply temporal convolution
            node_output = self.temporal_conv(node_data)

            # Apply ReLU
            node_output = F.relu(node_output)

            # Permute back to (B, T, hidden_channels)
            node_output = node_output.permute(0, 2, 1)

            temporal_out.append(node_output)

        # Stack along the node dimension (B, T, N, hidden_channels)
        temporal_out = torch.stack(temporal_out, dim=2)

        # Flatten for GRU processing (B*N, T, hidden_channels)
        x_flat = temporal_out.contiguous().view(
            B * self.num_nodes, T, self.hidden_channels
        )

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


class STGCNBlock(nn.Module):
    def __init__(
        self, in_channels, spatial_channels, out_channels, K, num_nodes, dilation
    ):
        super().__init__()
        self.temp1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            dilation=dilation,
        )
        self.cheb_conv = ChebConv(out_channels, spatial_channels, K)
        self.temp2 = nn.Conv2d(
            spatial_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            dilation=dilation,
        )
        # Replace LayerNorm with InstanceNorm2d which is designed for 4D tensors (B, C, H, W)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.num_nodes = num_nodes

    def forward(self, x, edge_index, edge_weight=None):
        # x: (B, C_in, N, T)
        x = self.temp1(x)  # → (B, C_out, N, T)
        x = F.relu(x)

        B, C, N, T = x.shape
        x_spatial = []
        for t in range(T):
            xt = x[:, :, :, t]  # (B, C, N)
            xt = xt.permute(0, 2, 1).contiguous()  # (B, N, C)
            xt = xt.reshape(B * N, C)  # (B*N, C)
            out = self.cheb_conv(xt, edge_index, edge_weight)
            out = out.view(B, N, -1).permute(0, 2, 1)  # (B, C_spatial, N)
            x_spatial.append(out)

        x = torch.stack(x_spatial, dim=-1)  # (B, C_spatial, N, T)
        x = self.temp2(x)
        # Apply normalization directly to the 4D tensor
        x = self.norm(x)  # (B, C, N, T)
        x = F.relu(x)
        return x  # (B, C, N, T)


class STGCN(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_vars,
        lags,
        horizon=24,
        K=2,
        spatial_channels=16,
        out_channels=64,
        dilation=1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.horizon = horizon  # horizon should be 24 as per your target

        # Define your layers
        self.block1 = STGCNBlock(
            num_vars, spatial_channels, out_channels, K, num_nodes, dilation
        )
        self.block1 = STGCNBlock(
            num_vars, spatial_channels, out_channels, K, num_nodes, dilation + 1
        )
        self.block1 = STGCNBlock(
            num_vars, spatial_channels, out_channels, K, num_nodes, dilation + 2
        )
        self.final_temporal = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)
        )

        # Adjust output layer to predict horizon=24 time steps (matching the target)
        self.output = nn.Conv2d(
            out_channels, horizon, kernel_size=(1, 1)
        )  # Output for 24 time steps

    def forward(self, x, edge_index, edge_weight=None):
        B, T, NF = x.shape  # B = batch size, T = time steps, NF = num_nodes * num_vars

        # Reshape input (B, T, N, F) for the network to process
        x = x.view(B, T, self.num_nodes, self.num_vars)  # (B, T, N, F)
        x = x.permute(0, 3, 2, 1)  # (B, F, N, T)

        # Apply the ST-GCN blocks (process the spatio-temporal data)
        x = self.block1(x, edge_index, edge_weight)

        # Apply the final temporal layer
        x = F.relu(self.final_temporal(x))

        # Output layer, generate predictions for horizon time steps
        x = self.output(x)  # (B, horizon, N, T)

        # Since we need (B, horizon, N) shape, we need to remove the time dimension
        # Take the last time step or mean across time dimension
        x = x.mean(dim=-1)  # (B, horizon, N)

        return x  # The final output has shape [B, horizon, N]
