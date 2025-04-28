import torch
import os
import sys

from logger import get_module_logger

from settings import (
    TRAIN_DATA_SPLIT,
    VAL_DATA_SPLIT,
    TEST_DATA_SPLIT,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    MAPPING_GRAPH_PATH
)

logger = get_module_logger("split_edges")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)


def split_edge_index(edge_index: torch.Tensor, 
                     train_split: float = TRAIN_DATA_SPLIT, 
                     val_split: float = VAL_DATA_SPLIT, 
                     test_split: float = TEST_DATA_SPLIT, 
                     seed: int = 42):
    """
    Splits edge_index into train, validation, and test sets based on specified ratios.

    Args:
        edge_index (torch.Tensor): Edge indices [2, num_edges].
        train_split (float): Proportion of edges for training.
        val_split (float): Proportion of edges for validation.
        test_split (float): Proportion of edges for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Train, validation, and test edge_index tensors.
    """
    logger.info("Starting edge splitting process...")

    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Ratios must sum to 1.0"

    num_edges = edge_index.size(1)
    torch.manual_seed(seed)

    logger.info(f"Total number of edges: {num_edges}")
    logger.info(f"Splitting ratios -> Train: {train_split}, Val: {val_split}, Test: {test_split}")

    # Shuffle the edges randomly
    permuted_indices = torch.randperm(num_edges)
    shuffled_edge_index = edge_index[:, permuted_indices]

    # Calculate split indices
    train_end = int(train_split * num_edges)
    val_end = int((train_split + val_split) * num_edges)

    # Perform the splits
    train_edge_index = shuffled_edge_index[:, :train_end]
    val_edge_index = shuffled_edge_index[:, train_end:val_end]
    test_edge_index = shuffled_edge_index[:, val_end:]

    logger.info(f"Edges split successfully: "
                f"Train {train_edge_index.size(1)}, "
                f"Val {val_edge_index.size(1)}, "
                f"Test {test_edge_index.size(1)}")
    
    save_splits(train_edge_index,val_edge_index, test_edge_index)

    return train_edge_index, val_edge_index, test_edge_index



def save_splits(train_edge_index: torch.Tensor, 
                val_edge_index: torch.Tensor, 
                test_edge_index: torch.Tensor):
    """
    Saves the split edge_index tensors to disk.

    Args:
        train_edge_index (torch.Tensor): Training edge indices.
        val_edge_index (torch.Tensor): Validation edge indices.
        test_edge_index (torch.Tensor): Testing edge indices.
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(TRAIN_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(VAL_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_DATA_PATH), exist_ok=True)

    logger.info(f"Saving train edges to {TRAIN_DATA_PATH}")
    torch.save(train_edge_index, TRAIN_DATA_PATH)

    logger.info(f"Saving validation edges to {VAL_DATA_PATH}")
    torch.save(val_edge_index, VAL_DATA_PATH)

    logger.info(f"Saving test edges to {TEST_DATA_PATH}")
    torch.save(test_edge_index, TEST_DATA_PATH)

    logger.info("All splits saved successfully.")
