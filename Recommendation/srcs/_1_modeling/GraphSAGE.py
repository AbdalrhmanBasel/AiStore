import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class RecommendationModel(nn.Module):
    def __init__(self, user_dim, item_dim, hidden_dim=256, embed_dim=64):
        super().__init__()
        self.user_encoder = GraphSAGE(user_dim, hidden_dim, embed_dim)
        self.item_encoder = GraphSAGE(item_dim, hidden_dim, embed_dim)
        self.scoring_head = nn.Linear(2*embed_dim, 1) 
        
    def forward(self, user_x, item_x, user_edge_index, item_edge_index):
        user_emb = self.user_encoder(user_x, user_edge_index)
        item_emb = self.item_encoder(item_x, item_edge_index)
        combined = torch.cat([user_emb, item_emb], dim=-1)
        return torch.sigmoid(self.scoring_head(combined))