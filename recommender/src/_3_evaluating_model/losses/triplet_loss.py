import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Compute Triplet Loss for fine-grained embedding optimization.
    
    Args:
        anchor (torch.Tensor): Embeddings of the anchor samples.
        positive (torch.Tensor): Embeddings of the positive samples.
        negative (torch.Tensor): Embeddings of the negative samples.
        margin (float): Margin for separating positive and negative distances.
    
    Returns:
        torch.Tensor: The mean triplet loss across all samples.
    """
    # Compute pairwise distances
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    
    loss = torch.relu(pos_dist - neg_dist + margin)
    return torch.mean(loss)