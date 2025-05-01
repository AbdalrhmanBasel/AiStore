import os
import sys
import torch
import pickle

from logger import get_module_logger
from torch_geometric.data import Data
from settings import (
    EDGE_INDEX_PATH,
    FEATURES_PATH,
    LABELS_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    MAPPING_GRAPH_PATH
)

logger = get_module_logger("load_saved_graph")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def load_saved_graph():
    """
    Loads the saved PyG graph data (edge_index, features, labels),
    train/val/test edge splits, and node ID mappings from disk.
    
    Returns:
        - graph (torch_geometric.data.Data): Full graph object.
        - mappings (dict): Dictionary containing user/item mappings.
        - train_eidx (Tensor): Edge indices for training.
        - val_eidx (Tensor): Edge indices for validation.
        - test_eidx (Tensor): Edge indices for testing.
    """

    logger.info("📂 Loading saved graph data and splits...")

    # Load edge_index, features, and labels
    edge_index = torch.load(EDGE_INDEX_PATH)
    x = torch.load(FEATURES_PATH)
    y = torch.load(LABELS_PATH)

    graph = Data(x=x, edge_index=edge_index, y=y)

    # Load edge splits
    train_eidx = torch.load(TRAIN_DATA_PATH)
    val_eidx = torch.load(VAL_DATA_PATH)
    test_eidx = torch.load(TEST_DATA_PATH)

    # Load mappings (user/item ID to index)
    with open(MAPPING_GRAPH_PATH, "rb") as f:
        mappings = pickle.load(f)

    logger.info("✅ Graph and associated data loaded successfully.")
    return graph, mappings, train_eidx, val_eidx, test_eidx
