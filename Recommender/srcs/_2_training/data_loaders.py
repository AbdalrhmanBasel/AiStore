from torch_geometric.loader import LinkNeighborLoader
import torch
from logger import get_module_logger
import os
import sys

logger = get_module_logger("data_loaders")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)

from settings import BATCH_SIZE

def create_data_loaders(graph, batch_size: int = BATCH_SIZE):
    """
    Creates train, validation, and test data loaders using LinkNeighborLoader.
    
    Args:
        graph (torch_geometric.data.Data): The full graph data with attached splits.
        batch_size (int): The batch size for training and validation loaders.

    Returns:
        Tuple of:
          - train_loader
          - val_loader
          - test_loader
    """
    logger.info(f"Creating data loaders with batch size: {batch_size} for graph with {graph.num_nodes} nodes and {graph.num_edges} edges.")
    
    try:
        # Training loader
        logger.info("Creating training data loader...")
        train_loader = LinkNeighborLoader(
            data=graph,
            edge_label_index=graph.train_edge_index,
            edge_label=torch.ones(graph.train_edge_index.size(1), dtype=torch.float),
            batch_size=batch_size,
            shuffle=True,
            num_neighbors=[10, 10],  # Adjust neighbor sizes per layer
        )
        logger.info(f"Training loader created with {len(train_loader)} batches.")

        # Validation loader
        logger.info("Creating validation data loader...")
        val_loader = LinkNeighborLoader(
            data=graph,
            edge_label_index=graph.val_edge_index,
            edge_label=torch.ones(graph.val_edge_index.size(1), dtype=torch.float),
            batch_size=batch_size,
            shuffle=False,
            num_neighbors=[10, 10],
        )
        logger.info(f"Validation loader created with {len(val_loader)} batches.")

        # Test loader
        logger.info("Creating test data loader...")
        test_loader = LinkNeighborLoader(
            data=graph,
            edge_label_index=graph.test_edge_index,
            edge_label=torch.ones(graph.test_edge_index.size(1), dtype=torch.float),
            batch_size=batch_size,
            shuffle=False,
            num_neighbors=[10, 10],
        )
        logger.info(f"Test loader created with {len(test_loader)} batches.")

    except Exception as e:
        logger.error(f"Error occurred while creating data loaders: {e}")
        raise e

    return train_loader, val_loader, test_loader
