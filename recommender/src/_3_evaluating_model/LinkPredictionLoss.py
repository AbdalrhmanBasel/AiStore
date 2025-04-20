import torch
import torch.nn as nn
import torch.nn.functional as F

class LinkPredictionLoss(nn.Module):
    """
    Link Prediction Loss for Graph Neural Networks (GNNs) based recommendation systems.
    This loss function assumes a binary classification task: predicting whether an edge exists
    between two nodes in the graph (e.g., whether a user interacted with a product).
    """

    def __init__(self):
        super(LinkPredictionLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        """
        Forward pass for the loss computation. The input scores are:
        - `pos_score`: Positive edge scores (real edges in the graph)
        - `neg_score`: Negative edge scores (non-existent edges in the graph)
        
        Returns:
            - loss: A scalar loss value to be minimized
        """

        # Positive samples are considered as actual edges (label = 1)
        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))

        # Negative samples are considered as non-existent edges (label = 0)
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))

        # Total loss is the sum of positive and negative losses
        total_loss = pos_loss + neg_loss

        return total_loss
