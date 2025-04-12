import torch
import torch.nn.functional as F
from torch.nn import Module, GRU, LSTM, Linear
from torch_geometric.nn import GCNConv


class TemporalGNN(Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=16,
        gcn_layers=2,  # Number of GCN layers is user-defined here
        rnn_layers=2,
        rnn_dropout=0.1,
        rnn_type="GRU",
        num_nodes=3,
    ):
        super(TemporalGNN, self).__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim

        # Dynamically create the graph convolution layers and store in a ModuleList
        self.convs = torch.nn.ModuleList()
        # First layer: from input_dim to hidden_dim
        self.convs.append(GCNConv(input_dim, hidden_dim))
        # Additional layers: keep the same hidden_dim
        for _ in range(gcn_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Temporal RNN: allow GRU or LSTM based on rnn_type
        self.rnn = getattr(torch.nn, rnn_type)(hidden_dim, hidden_dim, batch_first=True)

        # Final output layer to predict future values
        self.fc_out = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Process through the GCN layers dynamically
        # Use ReLU after the first layer, then feed each subsequent layer
        x = F.relu(self.convs[0](x, edge_index))
        for conv in self.convs[1:]:
            x = conv(x, edge_index)

        # Calculate batch_size from number of nodes
        batch_size = x.shape[0] // self.num_nodes

        # Reshape x for temporal processing: (batch_size, num_nodes, hidden_dim)
        x = x.view(batch_size, self.num_nodes, -1)

        # Apply the RNN to capture temporal dependencies
        x_rnn, _ = self.rnn(x)

        # Flatten for the final output layer: (batch_size * num_nodes, hidden_dim)
        x_flat = x_rnn.reshape(batch_size * self.num_nodes, -1)
        out = self.fc_out(x_flat)
        return out
