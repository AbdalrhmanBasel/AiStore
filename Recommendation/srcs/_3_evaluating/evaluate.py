import torch
import os
import sys
import torch.nn.functional as F
from logger import get_module_logger

logger = get_module_logger("evaluate")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)

def negative_sampling(
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int
) -> torch.Tensor:
    """
    Generate negative samples for link prediction.
    
    Args:
        pos_edge_index: LongTensor of shape [2, E] containing existing (positive) edges.
        num_nodes: Total number of nodes in the graph.
        num_neg_samples: Number of negative edges to sample.
    
    Returns:
        neg_edge_index: LongTensor of shape [2, num_neg_samples] containing sampled 
                        node pairs that are not in pos_edge_index.
    """
    # Build a quick set of positive pairs for O(1) membership tests
    pos_pairs = set(
        (u.item(), v.item()) for u, v in pos_edge_index.t()
    )
    
    neg_u = []
    neg_v = []
    while len(neg_u) < num_neg_samples:
        # sample random endpoints
        u = torch.randint(0, num_nodes, (1,), dtype=torch.long).item()
        v = torch.randint(0, num_nodes, (1,), dtype=torch.long).item()
        if (u, v) not in pos_pairs:
            neg_u.append(u)
            neg_v.append(v)
    
    return torch.tensor([neg_u, neg_v], dtype=torch.long)



# def evaluater():
#     logger.info("🔄 Starting evaluating process.")
#     logger.info("✅ Evaluating process completed.")


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    total_metric = 0.0
    for batch in loader:
        batch = batch.to(device)
        pos_idx = batch.edge_label_index
        neg_idx = negative_sampling(pos_idx, batch.num_nodes, pos_idx.size(1))

        # now our model returns (pos_scores, neg_scores)
        pos_scores, neg_scores = model(batch.x, batch.edge_index, pos_idx, neg_idx)

        # e.g. average BPR‐loss as proxy
        total_metric += -F.logsigmoid(pos_scores - neg_scores).mean().item()

    return total_metric / len(loader)


def compute_metrics(preds, targets):
    """
    Compute evaluation metrics (e.g., accuracy, AUC, etc.).
    Args:
        preds (torch.Tensor): The model predictions.
        targets (torch.Tensor): The ground truth labels.
    Returns:
        metrics (dict): A dictionary containing evaluation metrics.
    """
    # Here we can compute metrics such as accuracy or AUC
    accuracy = (preds.round() == targets).float().mean().item()
    return {'accuracy': accuracy}

