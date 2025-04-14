import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Module, Linear, Parameter, ModuleList
import torch.nn as nn


class AttentionGNN(Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, heads=8, dropout=0.6
    ):
        super(AttentionGNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.convs = ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )
        self.convs.append(
            GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout)
        )  # Output layer

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
