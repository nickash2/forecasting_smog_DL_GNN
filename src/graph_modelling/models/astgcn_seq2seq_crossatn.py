import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from .astgcn import SpatialAttentionLayer, TemporalAttentionLayer


class ASTGCN_Encoder(nn.Module):
    def __init__(
        self, num_nodes, num_vars, lags, block_channels, gru_channels, K, num_blocks
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags

        # Input embedding
        self.input_embedding = nn.Linear(num_vars, block_channels)

        # Spatial and temporal attention blocks
        self.blocks = nn.ModuleList()
        current_channels = block_channels
        for _ in range(num_blocks):
            block = nn.ModuleDict(
                {
                    "spatial_attn": SpatialAttentionLayer(
                        num_nodes, current_channels, block_channels
                    ),
                    "graph_conv": ChebConv(current_channels, block_channels, K=K),
                    "layer_norm1": nn.LayerNorm([lags, num_nodes, block_channels]),
                    "temporal_attn": TemporalAttentionLayer(
                        lags, block_channels, block_channels
                    ),
                    "temporal_gru": nn.GRU(
                        block_channels, gru_channels, batch_first=True
                    ),
                    "layer_norm2": nn.LayerNorm([lags, num_nodes, gru_channels]),
                }
            )
            self.blocks.append(block)
            current_channels = gru_channels

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Input (B, T, N*F)
            edge_index, edge_weight: Graph info
        Returns:
            Output (B, T, N, H)
        """
        B, T, NF = x.shape
        N, F_dim = self.num_nodes, self.num_vars
        assert T == self.lags and NF == N * F_dim

        # Reshape and embed
        x = x.view(B, T, N, F_dim)
        x = self.input_embedding(x)  # (B, T, N, block_channels)
        x = F.relu(x)  # Add ReLU activation

        # Process through blocks
        for block in self.blocks:
            # 1. Spatial attention + graph convolution
            x_sa = block["spatial_attn"](x)

            # Apply graph convolution per timestep
            x_gc_list = []
            for t in range(T):
                x_t = x_sa[:, t]  # (B, N, F)
                x_t_flat = x_t.reshape(B * N, -1)
                gc_out = block["graph_conv"](x_t_flat, edge_index, edge_weight)
                gc_out = gc_out.view(B, N, -1)
                x_gc_list.append(gc_out)

            x_gc = torch.stack(x_gc_list, dim=1)  # (B, T, N, H)
            x_gc = block["layer_norm1"](x_gc)

            # 2. Temporal attention + GRU
            x_ta = block["temporal_attn"](x_gc)

            # Apply GRU per node
            x_gru = x_ta.permute(0, 2, 1, 3)  # (B, N, T, H)
            x_gru = x_gru.reshape(B * N, T, -1)
            gru_out, _ = block["temporal_gru"](x_gru)
            gru_out = gru_out.view(B, N, T, -1).permute(0, 2, 1, 3)  # (B, T, N, H)

            x = block["layer_norm2"](gru_out)

        return x  # (B, T, N, H)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Projection layers for attention
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer normalization and dropout
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Initial query for the first timestep
        self.init_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.init_query)

    def forward(self, query, context, mask=None):
        """
        Args:
            query: (B*N, 1, H) or None for initialization
            context: (B*N, T, H)
            mask: Optional attention mask
        Returns:
            attn_output: (B*N, 1, H)
        """
        if query is None:
            # Using learned initial query for first timestep
            batch_size = context.size(0)
            query = self.init_query.repeat(batch_size, 1, 1)

        # Linear projections
        q = self.query_proj(query)  # (B*N, 1, H)
        k = self.key_proj(context)  # (B*N, T, H)
        v = self.value_proj(context)  # (B*N, T, H)

        # Scaled dot-product attention
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (
            self.hidden_dim**0.5
        )  # (B*N, 1, T)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_weights, dim=2)  # (B*N, 1, T)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights
        attn_output = torch.bmm(attn_weights, v)  # (B*N, 1, H)

        # Output projection
        attn_output = self.out_proj(attn_output)  # (B*N, 1, H)

        return attn_output, attn_weights


class ASTGCN_Decoder(nn.Module):
    def __init__(self, num_nodes, hidden_dim, forecast_horizon, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        # Cross-attention mechanism
        self.cross_attention = CrossAttention(hidden_dim, dropout)

        # GRU cell (not layer!) for recurrent processing
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # MLP for output projection after combining GRU output with attention
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Embedding for teacher forcing inputs
        self.input_embed = nn.Linear(1, hidden_dim)

    def forward(self, context, teacher_forcing_inputs=None, teacher_forcing_ratio=0.5):
        """
        Args:
            context: (B, T, N, hidden_dim) from encoder
            teacher_forcing_inputs: (B, forecast_horizon, N) optional (ground truth NO2 values)
            teacher_forcing_ratio: Probability of using teacher forcing
        Returns:
            outputs: (B, forecast_horizon, N)
        """
        B, T, N, H = context.shape
        device = context.device

        # Reshape context for attention: (B*N, T, H)
        context_flat = context.permute(0, 2, 1, 3).reshape(B * N, T, H)

        # Initialize hidden state with cross-attention
        init_attn_out, _ = self.cross_attention(None, context_flat)
        hidden = init_attn_out.view(B * N, H)  # (B*N, H)

        # Initialize first decoder input (zeros)
        decoder_input = torch.zeros(B * N, H, device=device)

        outputs = []
        attn_weights_all = []

        for t in range(self.forecast_horizon):
            # Get context vector via cross-attention
            decoder_input_expanded = decoder_input.unsqueeze(1)  # (B*N, 1, H)
            attn_out, attn_weights = self.cross_attention(
                decoder_input_expanded, context_flat
            )
            attn_out = attn_out.squeeze(1)  # (B*N, H)

            # Store attention weights for visualization if needed
            attn_weights_all.append(attn_weights)

            # Update hidden state using GRU cell
            hidden = self.gru_cell(decoder_input, hidden)  # (B*N, H)

            # Concatenate attention output and hidden state
            combined = torch.cat([hidden, attn_out], dim=1)  # (B*N, 2*H)

            # Generate prediction
            pred = self.output_mlp(combined)  # (B*N, 1)
            pred = pred.view(B, N)  # (B, N)
            outputs.append(pred)

            # Prepare next input - teacher forcing or using prediction
            if (
                teacher_forcing_inputs is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                # Use teacher forcing
                next_input = teacher_forcing_inputs[:, t, :].unsqueeze(-1)  # (B, N, 1)
                next_input = self.input_embed(next_input)  # (B, N, H)
                decoder_input = next_input.reshape(B * N, H)
            else:
                # Use prediction as next input
                pred_expanded = pred.unsqueeze(-1)  # (B, N, 1)
                next_input = self.input_embed(pred_expanded)  # (B, N, H)
                decoder_input = next_input.reshape(B * N, H)

        outputs = torch.stack(outputs, dim=1)  # (B, forecast_horizon, N)
        return outputs


class ASTGCN_Seq2Seq_CrossAttention(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_vars,
        lags,
        horizon,
        block_channels=32,
        gru_channels=32,
        K=1,
        num_blocks=1,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = ASTGCN_Encoder(
            num_nodes, num_vars, lags, block_channels, gru_channels, K, num_blocks
        )

        self.decoder = ASTGCN_Decoder(num_nodes, gru_channels, horizon, dropout)

    def forward(
        self,
        x,
        edge_index,
        edge_weight=None,
        teacher_forcing_inputs=None,
        teacher_forcing_ratio=0.5,
    ):
        """
        Args:
            x: (B, T, N*num_vars)
            edge_index: Graph connectivity
            edge_weight: Edge weights
            teacher_forcing_inputs: (B, horizon, N) optional
            teacher_forcing_ratio: Probability of using teacher forcing
        Returns:
            output: (B, horizon, N)
        """
        # Encode the input sequence
        context = self.encoder(x, edge_index, edge_weight)

        # Decode with cross-attention
        output = self.decoder(context, teacher_forcing_inputs, teacher_forcing_ratio)

        return output
