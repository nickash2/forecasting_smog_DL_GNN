import torch
import torch.nn as nn
import torch.nn.functional as F_func  # Add functional import for softmax
from .batched_graph_gru import BatchedGConvGRU


class AttentionGConvGRU(nn.Module):
    """
    Similar to BatchedGConvGRUIndex, but uses attention over the GRU's
    output sequence instead of just the final hidden state.
    """

    def __init__(
        self,
        num_nodes: int,
        num_vars: int,
        lags: int,
        hidden_channels: int = 32,
        horizon: int = 1,
        K: int = 2,
        normalization: str = "sym",
        bias: bool = True,
        batch_first: bool = True,
        # Attention specific parameters (optional)
        attention_mlp_hidden: int = 16,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.hidden_channels = hidden_channels
        self.horizon = horizon
        self.batch_first = batch_first

        self.gru = BatchedGConvGRU(
            in_channels=num_vars,
            out_channels=hidden_channels,
            K=K,
            normalization=normalization,
            bias=bias,
            batch_first=self.batch_first,
        )

        # Attention mechanism layers
        # Simple MLP to compute attention scores from hidden states
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_channels, attention_mlp_hidden),
            nn.Tanh(),
            nn.Linear(attention_mlp_hidden, 1),
        )

        # Prediction layer remains the same
        self.node_predict = nn.Linear(hidden_channels, horizon)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor,
        lambda_max: torch.Tensor = None,
        # H_0: Optional[torch.Tensor] = None, # Initial state for GRU
    ) -> torch.Tensor:
        """
        Args:
            x           (B, lags, N * F) or (lags, B, N * F)
            edge_index  (2, E)
            edge_weight (E,)
            lambda_max  (optional) Largest eigenvalue for ChebConv
        Returns:
            y_pred      (B, horizon, N)
        """
        # --- Input Reshaping (same as before) ---
        if self.batch_first:
            B, T, NF = x.size()
            N, F = self.num_nodes, self.num_vars
            assert T == self.lags and NF == N * F
            x_reshaped = x.view(B, T, N, F)
        else:
            T, B, NF = x.size()
            N, F = self.num_nodes, self.num_vars
            assert T == self.lags and NF == N * F
            x_reshaped = x.view(T, B, N, F)

        # --- Call BatchedGConvGRU (same as before) ---
        output_sequence, h_final = self.gru(
            X=x_reshaped,
            edge_index=edge_index,
            edge_weight=edge_weight,
            # H_0=H_0,
            lambda_max=lambda_max,
        )
        # output_sequence shape: (B, T, N, H) or (T, B, N, H)
        # h_final shape: (B, N, H)

        # --- Attention Mechanism ---
        if self.batch_first:
            # Input shape: (B, T, N, H)
            B, T, N, H = output_sequence.shape
            # Calculate attention scores
            attn_input = output_sequence.reshape(B * T * N, H)
            attn_scores = self.attention_mlp(attn_input)  # (B*T*N, 1)
            attn_scores = attn_scores.view(B, T, N, 1)  # Reshape: (B, T, N, 1)

            # Apply softmax over the time dimension (dim=1)
            attn_weights = F_func.softmax(attn_scores, dim=1)  # Shape: (B, T, N, 1)

            # Calculate context vector (weighted sum over time)
            # (B, T, N, H) * (B, T, N, 1) -> sum over T (dim=1) -> (B, N, H)
            context_vector = torch.sum(output_sequence * attn_weights, dim=1)
        else:
            # Input shape: (T, B, N, H)
            T, B, N, H = output_sequence.shape
            # Calculate attention scores
            attn_input = output_sequence.reshape(T * B * N, H)
            attn_scores = self.attention_mlp(attn_input)  # (T*B*N, 1)
            attn_scores = attn_scores.view(T, B, N, 1)  # Reshape: (T, B, N, 1)

            # Apply softmax over the time dimension (dim=0)
            attn_weights = F_func.softmax(attn_scores, dim=0)  # Shape: (T, B, N, 1)

            # Calculate context vector (weighted sum over time)
            # (T, B, N, H) * (T, B, N, 1) -> sum over T (dim=0) -> (B, N, H)
            context_vector = torch.sum(output_sequence * attn_weights, dim=0)

        # --- Use the context vector for prediction ---
        # context_vector shape: (B, N, hidden_channels)
        y = self.node_predict(
            context_vector
        )  # Input: (B, N, hidden), Output: (B, N, horizon)

        # Reorder to (B, horizon, N)
        y = y.permute(0, 2, 1).contiguous()
        return y
