import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, user_embeddings, item_embeddings, positive_item_indices, negative_item_indices):
        """
        Compute the BPR loss for a batch of users.

        :param user_embeddings: Tensor of user embeddings (shape: [num_users, embedding_dim])
        :param item_embeddings: Tensor of item embeddings (shape: [num_items, embedding_dim])
        :param positive_item_indices: Indices of positive items (shape: [batch_size])
        :param negative_item_indices: Indices of negative items (shape: [batch_size])
        :return: The BPR loss value
        """
        # Get the embeddings for positive and negative items for each user
        positive_item_embeddings = item_embeddings[positive_item_indices]  # Shape: [batch_size, embedding_dim]
        negative_item_embeddings = item_embeddings[negative_item_indices]  # Shape: [batch_size, embedding_dim]

        # Compute predicted scores for positive and negative items by taking the dot product
        positive_scores = torch.sum(user_embeddings * positive_item_embeddings, dim=1)  # [batch_size]
        negative_scores = torch.sum(user_embeddings * negative_item_embeddings, dim=1)  # [batch_size]

        # Compute the BPR loss: Log-sigmoid of the difference between positive and negative scores
        loss = -torch.mean(F.logsigmoid(positive_scores - negative_scores))
        return loss
