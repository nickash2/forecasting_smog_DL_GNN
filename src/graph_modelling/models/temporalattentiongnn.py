import torch
import torch.nn.functional as F
from torch.nn import Module, GRU, Linear
from torch_geometric.nn import GATConv


# 1) Temporal attention over RNN hidden states
class TemporalAttention(Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = Linear(hidden_dim, 1)

    def forward(self, h_seq):
        # h_seq: (batch*nodes, seq_len, hidden_dim)
        # compute unnormalized scores
        scores = self.score(h_seq).squeeze(-1)  # → (batch*nodes, seq_len)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # → (batch*nodes, seq_len, 1)
        # weighted sum over time
        return torch.sum(weights * h_seq, dim=1)  # → (batch*nodes, hidden_dim)


# 2) Full Spatio‑Temporal GAT + Temporal Attention model
class GATGRUGNN(Module):
    def __init__(
        self,
        input_features,  # features per node per timestep (e.g. 7)
        seq_len,  # history window (e.g. 72)
        forecast_horizon,  # how many hours ahead (e.g. 24)
        hidden_dim=32,
        gat_heads=4,
        gat_layers=2,
        rnn_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = input_features
        self.horizon = forecast_horizon
        self.output_dim = forecast_horizon

        # spatial attention: a stack of GATConv
        self.gats = torch.nn.ModuleList()
        # first layer: input_features→hidden_dim/heads
        self.gats.append(
            GATConv(
                input_features,
                hidden_dim // gat_heads,
                heads=gat_heads,
                dropout=dropout,
            )
        )
        for _ in range(gat_layers - 1):
            self.gats.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // gat_heads,
                    heads=gat_heads,
                    dropout=dropout,
                )
            )

        # temporal modeling: GRU over the GAT embeddings
        self.rnn = GRU(
            hidden_dim,
            hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout,
        )

        # temporal attention to pool the RNN outputs
        self.temporal_attn = TemporalAttention(hidden_dim)

        # final head: hidden_dim→forecast_horizon
        self.head = Linear(hidden_dim, forecast_horizon)

    def forward(self, data):
        """
        Handle either:
        - 3D input: data.x_seq: (num_nodes, seq_len, input_features)
        - 4D input: data.x_seq: (batch, num_nodes, seq_len, input_features)
        """
        x_seq = data.x_seq
        edge_index = data.edge_index

        # Handle 3D input (single sample case)
        if x_seq.dim() == 3:
            # Add batch dimension
            x_seq = x_seq.unsqueeze(0)  # [1, nodes, seq_len, features]

        # Now proceed with existing 4D logic
        B, N, T, F_in = x_seq.shape

        # Rest of the function remains the same
        spatial_emb = []
        for t in range(T):
            x_t = x_seq[:, :, t, :].reshape(B * N, F_in)
            for gat in self.gats:
                x_t = F.elu(gat(x_t, edge_index))
            spatial_emb.append(x_t.view(B, N, -1))

        spatial_seq = torch.stack(spatial_emb, dim=2)
        rnn_in = spatial_seq.view(B * N, T, -1)
        h_seq, _ = self.rnn(rnn_in)
        context = self.temporal_attn(h_seq)
        out = self.head(context)

        # If input was 3D, remove batch dimension from output
        if data.x_seq.dim() == 3:
            out = out.squeeze(0)

        return out
