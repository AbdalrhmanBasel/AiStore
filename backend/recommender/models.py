# recommender/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear

class HeteroGraphSAGE(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv1 = HeteroConv({
            ('user', 'rates', 'item'): SAGEConv((-1, -1), hidden_dim),
            ('item', 'rev_rates', 'user'): SAGEConv((-1, -1), hidden_dim)
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('user', 'rates', 'item'): SAGEConv(hidden_dim, hidden_dim),
            ('item', 'rev_rates', 'user'): SAGEConv(hidden_dim, hidden_dim)
        }, aggr='mean')
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_dict, edge_index_dict):
        # First conv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.bn1(x) for key, x in x_dict.items()}
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # Second conv layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: self.bn2(x) for key, x in x_dict.items()}
        return x_dict

class LinkPredictor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            Linear(hidden_dim, 1)
        )

    def forward(self, src, dst):
        return self.mlp(torch.cat([src, dst], dim=-1)).squeeze()