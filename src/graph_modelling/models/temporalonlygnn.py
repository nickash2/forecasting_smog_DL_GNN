import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleChebGRU(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_vars,
        lags,
        horizon,
        hidden_channels=32,
        gru_layers=1,
        dropout=0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.horizon = horizon
        self.hidden_channels = hidden_channels

        self.chebconv = ChebConv(
            in_channels=num_vars, out_channels=hidden_channels, K=1
        )

        if gru_layers > 1:
            dropout = dropout
        else:
            dropout = 0.0

        # GRU for temporal processing (node-wise)
        self.gru = nn.GRU(
            hidden_channels,
            hidden_channels,
            batch_first=True,
            num_layers=gru_layers,
            dropout=dropout,
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_channels, horizon)

    def forward(self, x, edge_index, edge_weight=None, lambda_max=None):
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
                xt, batched_edge_index, batched_edge_weight, lambda_max=lambda_max
            )  # [B*N, hidden]
            out = F.relu(out).view(B, self.num_nodes, self.hidden_channels)
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
        _, h_n = self.gru(cheb_seq)  # h_n: [1, B*N, hidden]
        # h_n is the hidden state for the last time step
        if self.gru.num_layers > 1:
            h_final = h_n[-1]

        else:
            # if only one layer, h_n is already the last hidden state
            h_final = h_n.squeeze(0)  # [B*N, hidden] 1 gru layer

        # 6) project and reshape to [B, horizon, N]
        y = (
            self.output_layer(h_final)  # [B*N, horizon]
            .view(B, self.num_nodes, self.horizon)
            .permute(0, 2, 1)
        )  # [B, horizon, N]

        return y
