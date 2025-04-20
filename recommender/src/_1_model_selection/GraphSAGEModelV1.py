import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F


class GraphSAGEModelV1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        """
        Initialize the GraphSAGE model for a recommendation system.

        Args:
            input_dim (int): The number of input features per node.
            hidden_dim (int): The number of features in the hidden layers.
            output_dim (int): The number of output features per node (embedding size).
            num_layers (int): Number of GraphSAGE layers (default: 2).
            dropout (float): Dropout rate for regularization (default: 0.5).
        """
        super(GraphSAGEModelV1, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, output_dim))

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, node_features, edge_indices):
        """
        Forward pass through the GraphSAGE model.

        Args:
            node_features (torch.Tensor): A tensor of node features.
            edge_indices (torch.Tensor): A tensor of edge indices defining the graph.

        Returns:
            torch.Tensor: The output node embeddings after the GraphSAGE layers.
        """
        x = node_features
        
        # Propagate through GraphSAGE layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_indices)
            
            # Apply ReLU activation
            if i != self.num_layers - 1: 
                x = F.relu(x)
            
            # Apply Dropout after each layer except the last
            if i != self.num_layers - 1:
                x = self.dropout_layer(x)
        
        return x



"""
USAGE EXAMPLE:
--------------

in_channels = 1
hidden_channels = 64  
out_channels = 32     
model = GraphSAGEModelV0(in_channels, hidden_channels, out_channels)
print(model)
"""