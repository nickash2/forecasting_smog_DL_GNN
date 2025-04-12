from torch.nn import Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Module
from torch.nn import ModuleList


class BasicGNN(Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16, num_gcn=2):
        super(BasicGNN, self).__init__()
        self.num_layers = num_gcn
        self.hidden_dim = hidden_dim
        self.convs = ModuleList(
            [
                GCNConv(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim if i < num_gcn - 1 else output_dim,
                )
                for i in range(num_gcn)
            ]
        )

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x
