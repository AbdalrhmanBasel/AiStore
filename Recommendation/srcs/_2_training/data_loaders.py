# from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import LinkNeighborLoader
import torch
from logger import get_module_logger
import os
import sys

logger = get_module_logger("trainer")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)

from settings import BATCH_SIZE

# In data_loaders.py

def create_data_loaders(graph, batch_size = BATCH_SIZE):
    train_loader = LinkNeighborLoader(
        data=graph,
        edge_label_index=graph.train_edge_index,
        edge_label=torch.ones(graph.train_edge_index.size(1)),  # Dummy labels
        batch_size=1024,
        shuffle=True,
        num_neighbors=[10, 10],
    )
    val_loader = LinkNeighborLoader(
        data=graph,
        edge_label_index=graph.val_edge_index,
        edge_label=torch.ones(graph.val_edge_index.size(1)),  # Dummy labels
        batch_size=1024,
        shuffle=False,
        num_neighbors=[10, 10],
    )
    return train_loader, val_loader
