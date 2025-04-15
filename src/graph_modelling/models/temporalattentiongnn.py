import torch
import torch.nn.functional as F
from torch.nn import Module, GRU, Linear, Parameter, ModuleList
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATGRUGNN(Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=16,
        gnn_layers=2,  # Number of attention (GAT) layers
        heads=8,
        rnn_layers=1,
        attention_dim=16,  # Dimensionality for the temporal attention
        dropout=0.6,
        num_nodes=3,  # Number of nodes per graph (or time steps)
        rnn_type="GRU",
    ):  # Could be extended to LSTM if needed
        super(GATGRUGNN, self).__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        # --- Attention-based Graph Convolutions ---
        self.att_gnn_convs = ModuleList()
        self.att_gnn_convs.append(
            GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        )
        # Additional layers: maintain hidden_dim, note the input dimension is hidden_dim*heads.
        for _ in range(gnn_layers - 1):
            self.att_gnn_convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )

        self.dropout = dropout
        # --- Temporal GRU ---
        # The GRU will take the aggregated node features as a sequence.
        # The input size to the GRU is the concatenated feature size (hidden_dim * heads)
        self.rnn = getattr(nn, rnn_type)(
            hidden_dim * heads,
            hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout,
        )

        # --- Temporal Attention Mechanism ---
        # We learn an attention vector to weight the GRU outputs over the "time" (or node) dimension.
        self.attn_linear = Linear(hidden_dim, attention_dim)
        self.attn_context = Parameter(torch.Tensor(attention_dim))
        # Initialize the attention context vector
        nn.init.uniform_(self.attn_context, -0.1, 0.1)

        # --- Final Output Layer ---
        self.fc_out = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # --- Spatial Attention using GATConv ---
        for conv in self.att_gnn_convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # --- Reshape for Temporal Processing ---
        batch_size = x.shape[0] // self.num_nodes
        x = x.view(
            batch_size, self.num_nodes, -1
        )  # Shape: [batch_size, num_nodes, feature_dimension]

        # --- Temporal GRU Processing ---
        rnn_out, _ = self.rnn(x)  # [batch_size, num_nodes, hidden_dim]

        # If you want node-level predictions, simply flatten the GRU output:
        batch_size, num_nodes, hidden_dim = rnn_out.shape
        rnn_out_flat = rnn_out.reshape(batch_size * num_nodes, hidden_dim)
        out = self.fc_out(rnn_out_flat)  # out: [batch_size * num_nodes, output_dim]
        return out
