import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear
from torch_geometric.nn import GCNConv


class BasicGNN(Module):
    def __init__(
        self,
        seq_len: int,  # e.g. 72
        num_features: int,  # e.g. 7
        forecast_horizon: int,  # e.g. 24
        hidden_dim: int = 16,
        num_gcn: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.forecast_horizon = forecast_horizon
        self.output_dim = forecast_horizon

        # Input dimension to GCN = seq_len * num_features
        in_dim = seq_len * num_features

        # Build GCN layers:
        #   first: in_dim -> hidden_dim
        #   intermediate: hidden_dim -> hidden_dim
        #   last: hidden_dim -> hidden_dim  (we’ll project to forecast_horizon below)
        self.convs = ModuleList()
        for i in range(num_gcn):
            in_ch = in_dim if i == 0 else hidden_dim
            out_ch = hidden_dim
            self.convs.append(GCNConv(in_ch, out_ch))

        # Final head: hidden_dim -> forecast_horizon
        self.head = Linear(hidden_dim, forecast_horizon)

    def forward(self, data):
        """
        Expects each `data` to have:
          - data.x_seq: (num_nodes, seq_len, num_features)
          - data.edge_index

        Returns:
          - (num_nodes, forecast_horizon)
        """
        x_seq = data.x_seq
        edge_index = data.edge_index

        # 1) Flatten each node’s history:
        #    (num_nodes, seq_len, num_features) -> (num_nodes, seq_len * num_features)
        num_nodes = x_seq.size(0)
        x = x_seq.view(num_nodes, self.seq_len * self.num_features)

        # 2) Run GCN stack:
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        # 3) Project to forecast horizon:
        out = self.head(x)  # (num_nodes, forecast_horizon)
        return out
