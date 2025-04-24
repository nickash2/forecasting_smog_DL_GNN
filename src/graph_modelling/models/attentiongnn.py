import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Module, Linear, Parameter, ModuleList, GRU
import torch.nn as nn


class AttentionGNN(Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, heads=8, dropout=0.6
    ):
        super(AttentionGNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # GAT layers for processing spatial relationships
        self.convs = ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )
        self.convs.append(
            GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout)
        )  # Output layer

        # Simple temporal aggregation (optional)
        self.temporal_aggregation = Linear(output_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        """
        Handle either:
        - Original format with data.x (for backward compatibility)
        - 3D input: data.x_seq: (num_nodes, seq_len, input_features)
        - 4D input: data.x_seq: (batch, num_nodes, seq_len, input_features)
        """
        # Check if we have temporal data
        if hasattr(data, "x_seq"):
            x_seq = data.x_seq
            edge_index = data.edge_index

            # Handle 3D input (single sample case)
            if x_seq.dim() == 3:
                # Add batch dimension
                x_seq = x_seq.unsqueeze(0)  # [1, nodes, seq_len, features]

            # Now process with 4D logic
            B, N, T, F_in = x_seq.shape

            # Process each time step with GAT
            temporal_outputs = []
            for t in range(T):
                # Get features for this timestep across all nodes & samples
                x_t = x_seq[:, :, t, :].reshape(B * N, F_in)

                # Process through GAT layers
                for i, conv in enumerate(self.convs):
                    x_t = conv(x_t, edge_index)
                    if i < len(self.convs) - 1:
                        x_t = F.elu(x_t)
                    x_t = F.dropout(x_t, p=self.dropout, training=self.training)

                # Store the output for this timestep
                temporal_outputs.append(x_t)

            # Simple aggregation of temporal outputs - last prediction is most important
            x = temporal_outputs[-1]

            # If input was 3D (single sample), remove the batch dimension from output
            if data.x_seq.dim() == 3:
                x = x.squeeze(0)

            return x

        else:
            # Original implementation for backward compatibility
            x, edge_index = data.x, data.edge_index

            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            return x
