import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def compute_loss(pos_out: torch.Tensor, neg_out: torch.Tensor) -> torch.Tensor:
    """
    Computes the loss for link prediction, including both positive and negative samples.
    
    Args:
    - pos_out: Tensor containing the output for positive samples (E_pos,).
    - neg_out: Tensor containing the output for negative samples (E_neg,).
    
    Returns:
    - Tensor representing the total loss.
    """
    pos_loss = -F.logsigmoid(pos_out).mean()
    neg_loss = -F.logsigmoid(-neg_out).mean()
    total_loss = pos_loss + neg_loss
    return total_loss



