import torch
import torch.nn as nn
from .batched_graph_gru import BatchedGConvGRU


class BatchedGConvGRUIndex(nn.Module):
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
        # Add batch_first argument to match BatchedGConvGRU
        batch_first: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags
        self.hidden_channels = hidden_channels
        self.horizon = horizon
        self.batch_first = batch_first  # Store batch_first

        self.gru = BatchedGConvGRU(
            in_channels=num_vars,
            out_channels=hidden_channels,
            K=K,
            normalization=normalization,
            bias=bias,
            batch_first=self.batch_first,  # Pass batch_first to the GRU layer
        )
        self.node_predict = nn.Linear(hidden_channels, horizon)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor,
        lambda_max: torch.Tensor = None,
        # Add H_0 if you want to pass an initial hidden state
        # H_0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x           (B, lags, N * F)  <- Assuming batch_first=True
            edge_index  (2, E)
            edge_weight (E,)
            lambda_max  (optional) Largest eigenvalue for ChebConv
            # H_0         (optional, B, N, hidden) Initial hidden state
        Returns:
            y_pred      (B, horizon, N)
        """
        if self.batch_first:
            # Input shape: (B, T, N*F)
            B, T, NF = x.size()
            N, F = self.num_nodes, self.num_vars
            assert T == self.lags, f"Expected {self.lags=} frames, got {T}"
            assert NF == N * F, f"Expected {N=}*{F=}, got {NF}"

            # Reshape input to (B, T, N, F) as expected by BatchedGConvGRU
            x_reshaped = x.view(B, T, N, F)
        else:
            # Input shape: (T, B, N*F) - Adapt if needed
            T, B, NF = x.size()
            N, F = self.num_nodes, self.num_vars
            assert T == self.lags, f"Expected {self.lags=} frames, got {T}"
            assert NF == N * F, f"Expected {N=}*{F=}, got {NF}"
            # Reshape input to (T, B, N, F)
            x_reshaped = x.view(T, B, N, F)

        # --- Call BatchedGConvGRU once for the whole sequence ---
        # It handles the internal time loop now.
        # We don't pass 'H' anymore. Can pass 'H_0' if needed.
        output_sequence, h_final = self.gru(
            X=x_reshaped,
            edge_index=edge_index,
            edge_weight=edge_weight,
            # H_0=H_0, # Pass initial state if provided
            lambda_max=lambda_max,
        )
        # output_sequence shape: (B, T, N, hidden) if batch_first=True
        # h_final shape: (B, N, hidden)

        # --- Use the final hidden state for prediction ---
        # h_final already has the correct shape (B, N, hidden_channels)
        y = self.node_predict(h_final)  # Input: (B, N, hidden), Output: (B, N, horizon)

        # Reorder to (B, horizon, N)
        y = y.permute(0, 2, 1).contiguous()
        return y
