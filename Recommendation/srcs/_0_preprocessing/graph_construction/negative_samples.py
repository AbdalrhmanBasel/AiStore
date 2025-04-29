from logger import get_module_logger
import torch
from torchmetrics.retrieval import RetrievalPrecision, RetrievalRecall, RetrievalNormalizedDCG

import os
import sys

from logger import get_module_logger

logger = get_module_logger("negative_samples")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)

# Move the generate_negative_samples function here before usage
def generate_negative_samples(pos_edge_index, num_neg_samples=5):
    """
    Generate negative samples for link prediction tasks.
    Args:
        pos_edge_index (torch.Tensor): The positive edge indices.
        num_neg_samples (int): Number of negative samples to generate per positive edge.
    Returns:
        neg_edge_index (torch.Tensor): The negative edge indices.
    """
    # Example negative sample generation: randomly sample non-existing edges.
    # This can be replaced with more sophisticated negative sampling methods.
    neg_edge_index = pos_edge_index.new_zeros((2, pos_edge_index.size(1) * num_neg_samples))
    return neg_edge_index
