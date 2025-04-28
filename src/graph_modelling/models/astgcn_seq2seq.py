from .astgcn import SpatioTemporalBlock
import torch
import torch.nn.functional as F_func
import torch.nn as nn


class ASTGCN_Encoder(nn.Module):
    def __init__(self, num_nodes, num_vars, lags, block_channels=32, gru_channels=32, K=2, num_blocks=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_vars = num_vars
        self.lags = lags

        self.input_embedding = nn.Linear(num_vars, block_channels)
        self.blocks = nn.ModuleList()
        
        current_channels = block_channels
        for _ in range(num_blocks):
            self.blocks.append(
                SpatioTemporalBlock(num_nodes, current_channels, block_channels, K, gru_channels)
            )
            current_channels = gru_channels  # Next block input

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: (B, T, N*num_vars)
        """
        B, T, NF = x.shape
        x = x.view(B, T, self.num_nodes, self.num_vars)
        x = self.input_embedding(x)
        
        for block in self.blocks:
            x = block(x, edge_index, edge_weight)
        
        # Final output: (B, T, N, hidden_dim)
        return x


class ASTGCN_Decoder(nn.Module):
    def __init__(self, num_nodes, hidden_dim, forecast_horizon):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)  # Predict 1 value (e.g., NO2) per node per step

    def forward(self, context, teacher_forcing_inputs=None, teacher_forcing_ratio=0.5):
        """
        context: (B, T, N, hidden_dim) from encoder
        teacher_forcing_inputs: (B, forecast_horizon, N) optional (ground truth NO2 values)
        """
        B, T, N, H = context.shape

        # Start with the last encoded step as initial hidden
        context_last = context[:, -1, :, :]  # (B, N, H)

        # Prepare GRU hidden states
        hidden = context_last.reshape(1, B * N, H)  # (1, B*N, H)

        outputs = []
        decoder_input = torch.zeros(B, N, H).to(context.device)  # initial input: zeros

        for t in range(self.forecast_horizon):
            # Run one step of GRU
            out, hidden = self.gru(decoder_input.view(B*N, 1, -1), hidden)
            out = out.view(B, N, H)

            pred = self.fc_out(out).squeeze(-1)  # (B, N)

            outputs.append(pred)

            if (teacher_forcing_inputs is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                # Teacher forcing: use true data
                decoder_input = teacher_forcing_inputs[:, t, :, None].repeat(1, 1, H)
            else:
                # Feed the model prediction
                decoder_input = out

        outputs = torch.stack(outputs, dim=1)  # (B, horizon, N)
        return outputs


class ASTGCN_Seq2Seq(nn.Module):
    def __init__(self, num_nodes, num_vars, lags, horizon, block_channels=32, gru_channels=32, K=1, num_blocks=1):
        super().__init__()
        self.encoder = ASTGCN_Encoder(num_nodes, num_vars, lags, block_channels, gru_channels, K, num_blocks)
        self.decoder = ASTGCN_Decoder(num_nodes, gru_channels, horizon)

    def forward(self, x, edge_index, edge_weight=None, teacher_forcing_inputs=None, teacher_forcing_ratio=0.5):
        """
        x: (B, T, N*num_vars)
        teacher_forcing_inputs: (B, horizon, N) optional
        """
        context = self.encoder(x, edge_index, edge_weight)
        output = self.decoder(context, teacher_forcing_inputs, teacher_forcing_ratio)
        return output
