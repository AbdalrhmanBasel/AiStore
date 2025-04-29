import os
import sys
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from logger import get_module_logger
from settings import IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NUM_LAYERS, DROPOUT

logger = get_module_logger("GraphSAGE")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        hidden_channels: int = HIDDEN_CHANNELS,
        out_channels: int = OUT_CHANNELS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT
    ):
        super().__init__()
        assert num_layers >= 2, "GraphSAGE requires at least two layers"
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        logger.info(f"Initialized GraphSAGE with {num_layers} layers | In: {in_channels}, "
                    f"Hidden: {hidden_channels}, Out: {out_channels}, Dropout: {dropout}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edge_index: torch.Tensor = None,
        neg_edge_index: torch.Tensor = None
    ) -> torch.Tensor:
        # 1) propagate features through all conv layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = self.relu(x)
                x = self.dropout(x)

        # 2) if no pos/neg passed, just return node embeddings
        if pos_edge_index is None or neg_edge_index is None:
            return x

        # 3) otherwise, compute dot-product scores for link prediction
        # pos scores
        u_pos = x[pos_edge_index[0]]
        v_pos = x[pos_edge_index[1]]
        pos_scores = (u_pos * v_pos).sum(dim=1)

        # neg scores
        u_neg = x[neg_edge_index[0]]
        v_neg = x[neg_edge_index[1]]
        neg_scores = (u_neg * v_neg).sum(dim=1)

        return pos_scores, neg_scores
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Extract node embeddings after performing the forward pass.
        """
        # Run through all convolution layers to get final node embeddings
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        return x
