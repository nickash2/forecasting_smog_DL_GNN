import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class BatchedGConvGRU(torch.nn.Module):
    """
    A batched version of the Graph Convolutional GRU that supports batch processing.
    Based on the original GConvGRU from torch_geometric_temporal.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: "sym").
        bias (bool, optional): If set to False, the layer will not learn
            an additive bias. (default: True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(BatchedGConvGRU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        """Create update gate parameters and layers."""
        self.conv_x_z = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_z = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):
        """Create reset gate parameters and layers."""
        self.conv_x_r = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_r = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):
        """Create candidate state parameters and layers."""
        self.conv_x_h = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_h = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        """Create all parameters and layers."""
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        """Initialize hidden state if not provided."""
        if H is None:
            H = torch.zeros(X.size(0), X.size(1), self.out_channels, device=X.device)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Forward pass through the Batched GConv GRU.

        Args:
            X: Input tensor of shape [batch_size, num_nodes, features]
            edge_index: Graph edge indices
            edge_weight: Edge weight tensor (optional)
            H: Hidden state tensor of shape [batch_size, num_nodes, out_features] (optional)
            lambda_max: Largest eigenvalue of Laplacian (optional)

        Returns:
            Updated hidden state of shape [batch_size, num_nodes, out_features]
        """
        # Get batch size and number of nodes
        batch_size, num_nodes, _ = X.size()

        # Initialize hidden state if not provided
        H = self._set_hidden_state(X, H)

        # Process each batch
        outputs = []
        for b in range(batch_size):
            # Get batch data
            X_b = X[b]  # Shape: [num_nodes, features]
            H_b = H[b]  # Shape: [num_nodes, out_features]

            # Update gate
            Z = self.conv_x_z(X_b, edge_index, edge_weight, lambda_max)
            Z = Z + self.conv_h_z(H_b, edge_index, edge_weight, lambda_max)
            Z = torch.sigmoid(Z)

            # Reset gate
            R = self.conv_x_r(X_b, edge_index, edge_weight, lambda_max)
            R = R + self.conv_h_r(H_b, edge_index, edge_weight, lambda_max)
            R = torch.sigmoid(R)

            # Candidate state
            H_tilde = self.conv_x_h(X_b, edge_index, edge_weight, lambda_max)
            H_tilde = H_tilde + self.conv_h_h(
                H_b * R, edge_index, edge_weight, lambda_max
            )
            H_tilde = torch.tanh(H_tilde)

            # New hidden state
            H_new = Z * H_b + (1 - Z) * H_tilde

            outputs.append(H_new)

        # Stack outputs
        return torch.stack(outputs)
