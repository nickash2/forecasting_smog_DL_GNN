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
        # 'data.x_seq' has shape: (num_nodes, window_size, num_features)
        x_seq = data.x_seq  # (num_nodes, window_size, num_features)
        edge_index = data.edge_index

        time_steps = x_seq.shape[1]
        gcn_outputs = []

        # Process each time step independently.
        for t in range(time_steps):
            # Get the feature matrix for time step t: (num_nodes, num_features)
            x_t = x_seq[:, t, :]
            x_gcn = F.relu(self.convs[0](x_t, edge_index))
            for conv in self.convs[1:]:
                x_gcn = conv(x_gcn, edge_index)
            gcn_outputs.append(x_gcn)

        # Stack along the time dimension: (num_nodes, time_steps, hidden_dim)
        x_gcn_seq = torch.stack(gcn_outputs, dim=1)

        # If processing a single graph, add batch dimension.
        x_gcn_seq = x_gcn_seq.unsqueeze(
            0
        )  # shape: (batch_size=1, num_nodes, time_steps, hidden_dim)

        batch_size = x_gcn_seq.shape[0]
        num_nodes = x_gcn_seq.shape[1]

        # Reshape to combine batch and node dims for RNN processing:
        # New shape: (batch_size * num_nodes, time_steps, hidden_dim)
        x_rnn_in = x_gcn_seq.reshape(batch_size * num_nodes, time_steps, -1)

        # Process with the RNN.
        x_rnn_out, _ = self.rnn(
            x_rnn_in
        )  # output shape: (batch_size*num_nodes, time_steps, hidden_dim)

        # Use only the last time step output.
        final_state = x_rnn_out[:, -1, :]  # shape: (batch_size*num_nodes, hidden_dim)
        out = self.fc_out(
            final_state
        )  # should now be (batch_size*num_nodes, output_dim)
        return out
