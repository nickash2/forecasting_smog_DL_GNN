import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, BatchNorm1d, LayerNorm
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

        # Increase dropout to prevent quick overfitting
        self.dropout = torch.nn.Dropout(0.4)

        # Add input layer normalization for more stable training
        self.input_norm = LayerNorm(in_dim)

        # Build GCN layers with batch normalization
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        for i in range(num_gcn):
            in_ch = in_dim if i == 0 else hidden_dim
            out_ch = hidden_dim
            self.convs.append(GCNConv(in_ch, out_ch))
            self.batch_norms.append(BatchNorm1d(out_ch))

        # Final head: hidden_dim -> forecast_horizon with additional regularization
        self.pre_head = Linear(hidden_dim, hidden_dim)
        self.head = Linear(hidden_dim, forecast_horizon)

        # Add weight initialization
        self.init_weights()

    def init_weights(self):
        """Initialize weights properly for stable training with more regularization"""
        for conv in self.convs:
            if hasattr(conv, "lin"):
                # Use xavier initialization with smaller values
                torch.nn.init.xavier_uniform_(conv.lin.weight, gain=0.5)
                if conv.lin.bias is not None:
                    torch.nn.init.zeros_(conv.lin.bias)

        # Initialize pre_head layer
        torch.nn.init.xavier_uniform_(self.pre_head.weight, gain=0.5)
        torch.nn.init.zeros_(self.pre_head.bias)

        # Initialize output layer with small weights to prevent quick convergence
        torch.nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        torch.nn.init.zeros_(self.head.bias)

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

        # 1) Flatten each node's history:
        #    (num_nodes, seq_len, num_features) -> (num_nodes, seq_len * num_features)
        num_nodes = x_seq.size(0)
        x = x_seq.view(num_nodes, self.seq_len * self.num_features)

        # Apply input normalization
        x = self.input_norm(x)

        # 2) Run GCN stack with regularization:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            # Use leaky ReLU for better gradient flow
            x = F.leaky_relu(x, negative_slope=0.1)
            x = self.dropout(x)

        # 3) Add an intermediate layer with activation and dropout
        x = self.pre_head(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        # 4) Project to forecast horizon:
        out = self.head(x)  # (num_nodes, forecast_horizon)

        # CRITICAL: Add sigmoid to bound outputs between 0-1
        # This is necessary since your targets are normalized to [0,1]
        out = torch.sigmoid(out)

        return out
