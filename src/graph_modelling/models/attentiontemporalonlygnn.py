import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch_geometric.nn import ChebConv, GCNConv, GATConv
from typing import Optional


# class TemporalAttention(nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         self.attn = nn.Linear(hidden_channels, 1)  # Attention mechanism

#     def forward(self, x):
#         # x: [B*N, T, F] where F = hidden_channels
#         attn_weights = self.attn(x)  # [B*N, T, 1]
#         attn_weights = F_func.softmax(
#             attn_weights, dim=1
#         )  # Softmax over the time dimension
#         weighted_x = torch.bmm(attn_weights.permute(0, 2, 1), x)  # [B*N, 1, F]
#         return weighted_x.squeeze(1)  # [B*N, F]


class TemporalAttention(nn.Module):
    def __init__(self, hidden_channels, dropout=0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 1),
            nn.Dropout(dropout),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        scores = self.attn(x) / self.temperature  # Control sharpness
        weights = F_func.softmax(scores, dim=1)
        return torch.bmm(weights.permute(0, 2, 1), x).squeeze(1)


class AttentionChebGRU(nn.Module):
    def __init__(
        self, num_nodes, num_vars, lags, horizon, hidden_channels=32, atn_dropout=0.0
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.horizon = horizon
        self.hidden_channels = hidden_channels

        # Single ChebConv layer
        self.chebconv = GATConv(
            in_channels=num_vars, out_channels=hidden_channels, heads=1, dropout=0.0
        )

        # GRU for temporal processing (node-wise)
        self.gru = nn.GRU(
            hidden_channels,
            hidden_channels,
            batch_first=True,
        )

        self.temporal_attn = TemporalAttention(hidden_channels, dropout=atn_dropout)

        # Output projection
        self.output_layer = nn.Linear(hidden_channels, horizon)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Tensor of shape [B, T, N*F] = [batch, lags, nodes * features]
            edge_index: [2, E] static adjacency for one graph of N nodes
            edge_weight: Optional [E] edge weights
        Returns:
            y: [B, horizon, N]
        """
        B, T, NF = x.shape
        assert NF == self.num_nodes * self.num_vars

        device = x.device
        # 1) build a batched edge_index so that each of the B graphs
        #    has its node indices offset by b*N
        batched_edge_indices = []
        for b in range(B):
            offset = b * self.num_nodes
            batched_edge_indices.append(edge_index + offset)
        batched_edge_index = torch.cat(batched_edge_indices, dim=1).to(device)
        # (if you have edge_weight, you’d similarly repeat it B times:
        batched_edge_weight = edge_weight.repeat(B)

        # 2) reshape input to [B, T, N, F]
        x = x.view(B, T, self.num_nodes, self.num_vars)

        cheb_outputs = []
        for t in range(T):
            # flatten batch & nodes → [B*N, F]
            xt = x[:, t, :, :].reshape(B * self.num_nodes, self.num_vars)
            # apply GCNConv over the big batched graph of B*N nodes
            out = self.chebconv(
                xt, batched_edge_index, batched_edge_weight
            )  # [B*N, hidden]
            out = F_func.relu(out).view(B, self.num_nodes, self.hidden_channels)
            cheb_outputs.append(out.unsqueeze(1))  # [B, 1, N, hidden]

        # 3) stack over time → [B, T, N, hidden]
        cheb_seq = torch.cat(cheb_outputs, dim=1)

        # 4) prep for GRU → [B*N, T, hidden]
        cheb_seq = (
            cheb_seq.permute(0, 2, 1, 3)  # [B, N, T, hidden]
            .contiguous()
            .view(B * self.num_nodes, T, self.hidden_channels)
        )

        # 5) GRU over time
        gru_out, h_n = self.gru(cheb_seq)  # h_n: [1, B*N, hidden]
        # h_n is the hidden state for the last time step
        # h_attn = self.temporal_attn(gru_out)  # [B*N, hidden]

        # disable hidden attn
        h_attn = h_n[-1]  # [B*N, hidden]

        y = (
            self.output_layer(h_attn)  # [B*N, horizon]
            .view(B, self.num_nodes, self.horizon)
            .permute(0, 2, 1)  # [B, horizon, N]
        )

        return y


class GATGRUModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_vars: int,
        lags: int,
        horizon: int,
        hidden_channels: int = 16,
        heads: int = 2,
        dropout: float = 0.2,
        # Add any parameters needed for TemporalAttention
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.horizon = horizon
        self.hidden_channels = hidden_channels
        self.heads = heads  # Store heads if needed later

        # 1) GAT to share info across nodes (takes raw features as input)
        self.gat = GATConv(
            in_channels=num_vars,  # Changed from hidden_channels to num_vars
            out_channels=hidden_channels // heads,
            heads=heads,
            concat=True,  # Output dimension will be heads * out_channels = hidden_channels
            dropout=dropout,
        )

        # 2) GRU for temporal processing after spatial context
        self.gru = nn.GRU(
            input_size=hidden_channels,  # Input is now GAT output
            hidden_size=hidden_channels,
            batch_first=True,
        )

        # 3) Temporal attention on the refined sequence
        self.temporal_attn = TemporalAttention(hidden_channels)

        # 4) Final projection to your forecast horizon
        self.out_proj = nn.Linear(hidden_channels, horizon)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight=None,  # edge_weight ignored by GATConv by default
    ) -> torch.Tensor:
        """
        x: [B, T, N * F]  where F = num_vars
        edge_index: [2, E] defining static connectivity among the N nodes
        returns y: [B, horizon, N]
        """
        B, T, NF = x.shape
        assert NF == self.num_nodes * self.num_vars
        device = x.device  # Get device from input tensor

        batched_edge_indices = []
        for b in range(B):
            offset = b * self.num_nodes
            batched_edge_indices.append(edge_index + offset)
        batched_edge_index = torch.cat(batched_edge_indices, dim=1).to(device)

        # reshape into per-node sequences for timestep processing
        x = x.view(B, T, self.num_nodes, self.num_vars)  # [B, T, N, F]

        # 1) spatial encoding with GAT at each timestep
        gat_seq = []
        for t in range(T):
            # get features at this timestep: [B, N, F]
            x_t = x[:, t, :, :]
            # reshape for GAT: [B*N, F]
            x_t = x_t.reshape(B * self.num_nodes, self.num_vars)

            # apply GAT using the batched edge_index
            h_t = self.gat(x_t, batched_edge_index)  # [B*N, hidden]
            h_t = F_func.relu(h_t)  # Apply activation

            # collect as sequence: [B*N, 1, hidden]
            gat_seq.append(h_t.unsqueeze(1))

        # 2) concatenate to form temporal sequence: [B*N, T, hidden]
        gat_seq = torch.cat(gat_seq, dim=1)

        # 3) process with GRU for temporal patterns
        gru_out, _ = self.gru(gat_seq)  # [B*N, T, hidden]

        # 4) temporal attention on GRU output
        h_attn = self.temporal_attn(gru_out)  # [B*N, hidden]

        # 5) project and reshape to [B, horizon, N]
        y = self.out_proj(h_attn)  # [B*N, horizon]
        y = y.view(B, self.num_nodes, self.horizon).permute(0, 2, 1)  # [B, horizon, N]

        return y
