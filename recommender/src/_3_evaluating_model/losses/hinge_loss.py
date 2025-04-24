import torch

def compute_hinge_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Compute the hinge loss for ranking tasks.

    The hinge loss formula is:
        L = max(0, margin - (positive_scores - negative_scores))

    This ensures that the scores for positive samples are higher than those for negative 
    samples by at least the specified margin.

    Args:
        pos_scores (torch.Tensor): Scores for positive samples.
        neg_scores (torch.Tensor): Scores for negative samples.
        margin (float): The margin by which positive scores should exceed negative scores.
                        Default is 1.0.

    Returns:
        torch.Tensor: The mean hinge loss across all samples.
    """
    score_difference = pos_scores - neg_scores
    loss = torch.relu(margin - score_difference)
    return torch.mean(loss)


if __name__ == "__main__":
    pos_scores = torch.tensor([1.2, 0.8, 1.5])
    neg_scores = torch.tensor([0.5, 0.9, 1.0])

    loss = compute_hinge_loss(pos_scores, neg_scores, margin=1.0)
    print(f"Hinge Loss: {loss.item():.4f}")