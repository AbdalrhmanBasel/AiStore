# import os
# import sys

# import torch
# from torch_geometric.nn import SAGEConv
# import torch.nn.functional as F

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# sys.path.append(PROJECT_ROOT)

# from settings import HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_RATE


# class GraphSAGEModelV0(torch.nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
#                  output_dim: int = OUTPUT_DIM, num_layers: int = NUM_LAYERS,
#                  dropout: float = DROPOUT_RATE):
#         super(GraphSAGEModelV0, self).__init__()

#         # Validate the number of layers
#         if num_layers < 2:
#             raise ValueError("Number of layers must be at least 2.")

#         # Model parameters
#         self.num_layers = num_layers
#         self.dropout = dropout

#         # Define GraphSAGE layers
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(SAGEConv(input_dim, hidden_dim))  # Input layer
#         for _ in range(num_layers - 2):
#             self.convs.append(SAGEConv(hidden_dim, hidden_dim))  # Hidden layers
#         self.convs.append(SAGEConv(hidden_dim, output_dim))  # Output layer

#         # Dropout layer
#         self.dropout_layer = torch.nn.Dropout(dropout)

#         # Add a decoder for edge scoring
#         self.decoder = torch.nn.Linear(output_dim, 1)  # Outputs a single scalar score per edge

#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
#         # Propagate through GraphSAGE layers
#         for i in range(self.num_layers):
#             x = self.convs[i](x, edge_index)
#             if i != self.num_layers - 1:
#                 x = F.relu(x)
#                 x = self.dropout_layer(x)

#         # Decode edge scores using the decoder
#         return self.decoder(x)  # Output edge-level scores

# if __name__ == "__main__":
#     """
#     Example usage of the GraphSAGEModelV0 class.

#     Creates a model with specified input, hidden, and output dimensions,
#     and prints the model architecture.
#     """
#     in_channels = 16      # Example: 16 input features per node
#     hidden_channels = 64  # Example: 64 hidden features
#     out_channels = 32     # Example: 32 output features (embedding size)

#     model = GraphSAGEModelV0(in_channels, hidden_channels, out_channels)
#     print(model)


import os
import sys

import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)

from settings import HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_RATE


class GraphSAGEModelV0(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 output_dim: int = OUTPUT_DIM, num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT_RATE):
        super(GraphSAGEModelV0, self).__init__()

        # Validate the number of layers
        if num_layers < 2:
            raise ValueError("Number of layers must be at least 2.")

        # Model parameters
        self.num_layers = num_layers
        self.dropout = dropout

        # Define GraphSAGE layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))  # Input layer
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))  # Hidden layers
        self.convs.append(SAGEConv(hidden_dim, output_dim))  # Output layer

        # Dropout layer
        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Propagate through GraphSAGE layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)

        # Return node embeddings (no decoder, scoring is done externally)
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

    model = GraphSAGEModelV0(in_channels, hidden_channels, out_channels)
    print(model)
