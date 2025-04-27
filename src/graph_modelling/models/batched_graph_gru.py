import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from typing import Optional, List, Tuple  # Added for type hinting


class BatchedGConvGRU(torch.nn.Module):
    """
    A batched version of the Graph Convolutional GRU that processes sequences internally.
    Based on the original GConvGRU from torch_geometric_temporal.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: "sym").
        bias (bool, optional): If set to False, the layer will not learn
            an additive bias. (default: True)
        batch_first (bool, optional): If True, then the input and output tensors
            are provided as (batch, seq, node, feature). Default: True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
        batch_first: bool = True,  # Added batch_first argument
    ):
        super(BatchedGConvGRU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self.batch_first = batch_first  # Store batch_first
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

    def _compute_gru_step(
        self, X_t_flat, H_prev_flat, edge_index, edge_weight, lambda_max
    ):
        """Computes one step of GRU logic on flattened tensors."""
        # --- Update Gate ---
        update_gate_x = self.conv_x_z(
            X_t_flat, edge_index, edge_weight, batch=None, lambda_max=lambda_max
        )
        update_gate_h = self.conv_h_z(
            H_prev_flat, edge_index, edge_weight, batch=None, lambda_max=lambda_max
        )
        Z_flat = torch.sigmoid(update_gate_x + update_gate_h)

        # --- Reset Gate ---
        reset_gate_x = self.conv_x_r(
            X_t_flat, edge_index, edge_weight, batch=None, lambda_max=lambda_max
        )
        reset_gate_h = self.conv_h_r(
            H_prev_flat, edge_index, edge_weight, batch=None, lambda_max=lambda_max
        )
        R_flat = torch.sigmoid(reset_gate_x + reset_gate_h)

        # --- Candidate State ---
        candidate_state_x = self.conv_x_h(
            X_t_flat, edge_index, edge_weight, batch=None, lambda_max=lambda_max
        )
        candidate_state_h = self.conv_h_h(
            H_prev_flat * R_flat,
            edge_index,
            edge_weight,
            batch=None,
            lambda_max=lambda_max,
        )
        H_tilde_flat = torch.tanh(candidate_state_x + candidate_state_h)

        # --- New Hidden State ---
        H_new_flat = Z_flat * H_prev_flat + (1 - Z_flat) * H_tilde_flat
        return H_new_flat

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.FloatTensor] = None,
        H_0: Optional[torch.FloatTensor] = None,  # Initial hidden state
        lambda_max: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:  # Return sequence and final state
        """
        Forward pass through the Batched GConv GRU processing the whole sequence.

        Args:
            X: Input tensor. Shape depends on batch_first:
               (batch_size, seq_length, num_nodes, in_channels) if batch_first=True
               (seq_length, batch_size, num_nodes, in_channels) if batch_first=False
            edge_index: Graph edge indices of shape [2, num_edges].
                       Assumed to be the same for all graphs in the batch.
            edge_weight: Edge weight tensor of shape [num_edges] (optional).
                       Assumed to be the same for all graphs in the batch.
            H_0: Initial hidden state tensor of shape [batch_size, num_nodes, out_channels] (optional).
                 Defaults to zeros if not provided.
            lambda_max: Largest eigenvalue of Laplacian (optional).

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]:
            - output_sequence: Tensor containing the output features (hidden state) for each time step.
                               Shape depends on batch_first:
                               (batch_size, seq_length, num_nodes, out_channels) if batch_first=True
                               (seq_length, batch_size, num_nodes, out_channels) if batch_first=False
            - H_final: Tensor containing the final hidden state for the sequence.
                       Shape: (batch_size, num_nodes, out_channels)
        """
        if self.batch_first:
            # Input: (B, T, N, F_in)
            batch_size, seq_length, num_nodes, _ = X.size()
        else:
            # Input: (T, B, N, F_in)
            seq_length, batch_size, num_nodes, _ = X.size()
            # Temporarily permute to (B, T, N, F_in) for easier processing
            X = X.permute(1, 0, 2, 3)

        # Initialize hidden state if not provided
        if H_0 is None:
            H_t = torch.zeros(batch_size, num_nodes, self.out_channels, device=X.device)
        else:
            H_t = H_0

        outputs: List[torch.FloatTensor] = []  # To store hidden states of each step

        # Loop through time steps
        for t in range(seq_length):
            # Get input for current time step: (B, N, F_in)
            X_t = X[:, t, :, :]

            # Reshape for PyG compatibility: (B, N, F) -> (B*N, F)
            X_t_flat = X_t.reshape(-1, self.in_channels)
            H_t_flat = H_t.reshape(-1, self.out_channels)  # Previous hidden state

            # Compute one GRU step using the flattened tensors
            H_next_flat = self._compute_gru_step(
                X_t_flat, H_t_flat, edge_index, edge_weight, lambda_max
            )

            # Reshape back to batched format: (B*N, F_out) -> (B, N, F_out)
            H_t = H_next_flat.reshape(
                batch_size, num_nodes, self.out_channels
            )  # Update H_t for next step
            outputs.append(H_t)

        # Stack outputs along the time dimension
        output_sequence = torch.stack(
            outputs, dim=1 if self.batch_first else 0
        )  # dim=1 for (B, T, N, F), dim=0 for (T, B, N, F)

        # H_t already holds the final hidden state
        H_final = H_t

        return output_sequence, H_final
