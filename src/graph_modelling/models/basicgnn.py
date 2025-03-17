import torch.nn.Module
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class BasicGNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
