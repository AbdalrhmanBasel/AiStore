import torch
import torch.nn as nn

def compute_margin_ranking_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Compute Margin Ranking Loss for ranking tasks.
    
    Args:
        pos_scores (torch.Tensor): Scores for positive samples.
        neg_scores (torch.Tensor): Scores for negative samples.
        margin (float): Margin for separating positive and negative scores.
    
    Returns:
        torch.Tensor: The mean margin ranking loss across all samples.
    """
    loss_fn = nn.MarginRankingLoss(margin=margin)
    target = torch.ones_like(pos_scores)  # Target is always 1 for ranking tasks
    return loss_fn(pos_scores, neg_scores, target)