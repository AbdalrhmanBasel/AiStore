import os
import sys

import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)

from settings import HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_RATE

class GraphSAGEModelV0(torch.nn.Module):
    """
    A GraphSAGE-based model for recommendation systems.

    This model uses multiple stacked GraphSAGE layers to generate node embeddings.
    It supports dropout, batch normalization, and residual connections for regularization.
    """

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 output_dim: int = OUTPUT_DIM, num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT_RATE, activation=F.relu, use_batch_norm: bool = True,
                 use_residual: bool = False):
        """
        Initialize the GraphSAGE model.

        Args:
            input_dim (int): The number of input features per node.
            hidden_dim (int): The number of features in the hidden layers.
            output_dim (int): The number of output features per node (embedding size).
            num_layers (int): Number of GraphSAGE layers.
            dropout (float): Dropout rate for regularization.
            activation (callable): Activation function to use between layers.
            use_batch_norm (bool): Whether to use batch normalization.
            use_residual (bool): Whether to use residual connections.
        """
        super(GraphSAGEModelV0, self).__init__()

        # Validate the number of layers
        if num_layers < 2:
            raise ValueError("Number of layers must be at least 2.")

        # Model parameters
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        # Define GraphSAGE layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))  # Input layer
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))  # Hidden layers
        self.convs.append(SAGEConv(hidden_dim, output_dim))  # Output layer

        # Optional batch normalization layers
        if use_batch_norm:
            self.bns = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)]
            )
        else:
            self.bns = None

        # Dropout layer
        self.dropout_layer = torch.nn.Dropout(dropout)

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Kaiming initialization."""
        for conv in self.convs:
            conv.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GraphSAGE model.

        Args:
            x (torch.Tensor): A tensor of node features.
            edge_index (torch.Tensor): A tensor of edge indices defining the graph.

        Returns:
            torch.Tensor: The output node embeddings after the GraphSAGE layers.
        """
        residual = x  # Store input for residual connections

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)

            # Apply batch normalization (except for the last layer)
            if self.use_batch_norm and i != self.num_layers - 1:
                x = self.bns[i](x)

            # Apply activation (except for the last layer)
            if i != self.num_layers - 1:
                x = self.activation(x)

            # Apply residual connection (if enabled)
            if self.use_residual and i != 0 and i != self.num_layers - 1:
                x = x + residual
                residual = x  # Update residual for the next layer

            # Apply dropout (except for the last layer)
            if i != self.num_layers - 1:
                x = self.dropout_layer(x)

        return x


if __name__ == "__main__":
    """
    Example usage of the GraphSAGEModelV0 class.

    Creates a model with specified input, hidden, and output dimensions,
    and prints the model architecture.
    """
    in_channels = 16      # Example: 16 input features per node
    hidden_channels = 64  # Example: 64 hidden features
    out_channels = 32     # Example: 32 output features (embedding size)

    model = GraphSAGEModelV0(in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, use_batch_norm=True, use_residual=True)
    print(model)