from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch


class TemporalGNN(Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=16,
        num_layers=2,
        rnn_type="GRU",
        num_nodes=3,
    ):
        super(TemporalGNN, self).__init__()
        self.num_nodes = num_nodes

        # Graph Convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Temporal RNN (GRU or LSTM)
        self.rnn = getattr(torch.nn, rnn_type)(hidden_dim, hidden_dim, batch_first=True)

        # Final output layer to predict future pollution levels
        self.fc_out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Spatial (graph) convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)  # Output shape: (batch_size * 3, output_dim)

        # We expect x to be of shape (batch_size * 3, num_features)
        batch_size = (
            x.shape[0] // self.num_nodes
        )  # Divide by 3 because you have 3 nodes in your graph

        # Reshape to (batch_size, 3, num_features) for temporal processing
        x = x.view(
            batch_size, self.num_nodes, -1
        )  # Reshapes x to (batch_size, 3, output_dim)

        # Temporal RNN processing (GRU or LSTM)
        x_rnn, _ = self.rnn(x)  # Apply RNN to learn temporal patterns

        out = self.fc_out(
            x_rnn.reshape(
                batch_size * self.num_nodes, -1
            )  # Flatten the batch and nodes using reshape
        )  # Expected shape: (batch_size * 3, output_dim)
        return out
