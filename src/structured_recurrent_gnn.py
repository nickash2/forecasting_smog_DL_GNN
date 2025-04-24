import torch
import torch.nn.functional as F
from batched_graph_gru import BatchedGConvGRU


class StructuredRecurrentGNN(torch.nn.Module):
    def __init__(
        self, node_features, num_vars, num_lags, hidden_channels=32, out_channels=1, k=2
    ):
        super(StructuredRecurrentGNN, self).__init__()

        self.node_features = node_features
        self.num_vars = num_vars
        self.num_lags = num_lags
        self.num_nodes = 3  # Amsterdam, Rotterdam, Utrecht

        # Adjust hidden channels to be divisible by num_vars
        self.hidden_per_var = hidden_channels // num_vars
        self.total_hidden = (
            self.hidden_per_var * num_vars * self.num_nodes
        )  # Include num_nodes

        # Each variable gets its own recurrent GNN module
        self.var_recurrent = torch.nn.ModuleList(
            [
                BatchedGConvGRU(
                    in_channels=num_lags, out_channels=self.hidden_per_var, K=k
                )
                for _ in range(num_vars)
            ]
        )

        # Output layers - with corrected dimensions
        self.combine = torch.nn.Linear(self.total_hidden, self.total_hidden // 2)
        self.final = torch.nn.Linear(self.total_hidden // 2, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        batch_size, n_lags, features = x.size()

        # Reshape to separate variables
        # From [batch_size, n_lags, num_nodes * num_vars]
        # To [batch_size, num_vars, num_nodes, n_lags]
        x_reshaped = x.reshape(batch_size, n_lags, self.num_nodes, self.num_vars)
        x_reshaped = x_reshaped.permute(0, 3, 2, 1)  # [batch, vars, nodes, lags]

        # Process each variable separately
        var_outputs = []
        for i in range(self.num_vars):
            # Extract this variable's data [batch, nodes, lags]
            var_data = x_reshaped[:, i, :, :]

            # Process with BatchedGConvGRU
            var_output = self.var_recurrent[i](var_data, edge_index, edge_weight)
            var_outputs.append(var_output.reshape(batch_size, -1))

        # Rest of method remains the same
        combined = torch.cat(var_outputs, dim=1)
        h = F.relu(combined)
        h = self.combine(h)
        h = F.relu(h)
        output = self.final(h)

        return output
