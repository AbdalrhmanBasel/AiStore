import os
import sys
import torch
from logger import get_module_logger
from torch_geometric.data import Data
from settings import (
    SAVED_MODEL_DIR, 
    EDGE_INDEX_PATH, 
    FEATURES_PATH, 
    LABELS_PATH,
)

# Initialize the logger
logger = get_module_logger("load_model")

# Set the project root directory and include it in the system path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)

def load_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, checkpoint_path: str = SAVED_MODEL_DIR, device: str = "cpu"):
    """
    Load the model (and optionally optimizer and epoch) from a checkpoint.

    Args:
        model (torch.nn.Module): The model to load state_dict into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state_dict into.
        checkpoint_path (str): Path to the checkpoint to load.
        device (str): Device to load the model on (e.g., "cpu", "cuda").

    Returns:
        model (torch.nn.Module): The model with loaded state_dict.
        optimizer (torch.optim.Optimizer, optional): The optimizer with loaded state_dict, if provided.
        epoch (int, optional): The epoch number, if saved.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state (optional)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load epoch (optional)
    epoch = checkpoint.get("epoch", None)

    logger.info(f"🔄 Model loaded from {checkpoint_path}.")
    if epoch is not None:
        logger.info(f"🔄 Resumed from epoch {epoch}.")
    
    return model, optimizer, epoch


def load_graph_data():
    """
    Loads the graph data (node features and edge indices).
    Returns a graph object.
    """
    logger.info("➡️  Loading graph data...")
    
    # Load edge index, node features, and labels from paths defined in settings.py
    edge_index = torch.load(EDGE_INDEX_PATH)
    node_features = torch.load(FEATURES_PATH)
    labels = torch.load(LABELS_PATH)
    
    # Create a graph object using PyTorch Geometric Data
    graph = Data(x=node_features, edge_index=edge_index, y=labels)
    
    # Ensure the graph object contains 'x' for node features and 'edge_index' for edges
    assert hasattr(graph, 'x'), "Graph object must have 'x' for node features."
    assert hasattr(graph, 'edge_index'), "Graph object must have 'edge_index' for edge indices."
    
    logger.info("✅ Graph data loaded successfully.")
    return graph


