from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


def precision_at_k(y_true, y_score, k):
    """Batch-compatible Precision@k"""
    top_k_indices = np.argsort(y_score)[-k:]
    return np.sum(y_true[top_k_indices]) / k


def recall_at_k(y_true, y_score, k):
    """Batch-compatible Recall@k"""
    top_k_indices = np.argsort(y_score)[-k:]
    relevant = np.sum(y_true)
    return np.sum(y_true[top_k_indices]) / relevant if relevant != 0 else 0.0


def mrr(y_true, y_score):
    """Mean Reciprocal Rank"""
    order = np.argsort(y_score)[::-1]
    for rank, idx in enumerate(order):
        if y_true[idx] == 1:
            return 1.0 / (rank + 1)
    return 0.0


def compute_ndcg_at_k(score_pairs: np.ndarray, k: int = 10) -> float:
    """
    Compute average NDCG@k for (pos, neg) score pairs.
    """
    ndcgs = []  # Corrected the variable name to ndcgs
    for pos, neg in score_pairs:
        scores = [pos, neg]
        labels = [1, 0]
        rank = np.argsort(scores)[::-1]
        dcg = (2 ** labels[rank[0]] - 1) / np.log2(2) + (2 ** labels[rank[1]] - 1) / np.log2(3)
        idcg = (2 ** 1 - 1) / np.log2(2)
        ndcgs.append(dcg / idcg)  # Appending to ndcgs list
    return float(np.mean(ndcgs))  # Ensure it returns the average of ndcgs



@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: str = "cpu", topk: int = 10) -> dict:
    """
    Evaluate GNN model for link prediction with robust metrics.

    Args:
        model (torch.nn.Module): The GNN model.
        loader: PyG DataLoader for validation/test data.
        device (str): 'cpu' or 'cuda'.
        topk (int): Top-k threshold for ranking metrics.

    Returns:
        dict: Dictionary with BPR loss, AUC, AP, NDCG, Precision@k, Recall@k, MRR.
    """
    model.eval()
    all_pos_scores = []
    all_neg_scores = []
    bpr_losses = []
    all_y_true = []
    all_y_score = []

    for batch in loader:
        batch = batch.to(device)

        pos_idx = batch.edge_label_index
        neg_idx = negative_sampling(
            edge_index=pos_idx,
            num_nodes=batch.num_nodes,
            num_neg_samples=pos_idx.size(1)
        )

        pos_scores, neg_scores = model(batch.x, batch.edge_index, pos_idx, neg_idx)

        # BPR Loss
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean().item()
        bpr_losses.append(bpr_loss)

        # Accumulate for global metrics
        pos_np = pos_scores.cpu().numpy()
        neg_np = neg_scores.cpu().numpy()

        all_pos_scores.extend(pos_np)
        all_neg_scores.extend(neg_np)

        all_y_true.extend([1] * len(pos_np) + [0] * len(neg_np))
        all_y_score.extend(np.concatenate([pos_np, neg_np]))

    y_true = np.array(all_y_true)
    y_score = np.array(all_y_score)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    score_pairs = np.stack([all_pos_scores, all_neg_scores], axis=1)
    ndcg = compute_ndcg_at_k(score_pairs, k=topk)
    prec = precision_at_k(y_true, y_score, k=topk)
    rec = recall_at_k(y_true, y_score, k=topk)
    mrr_score = mrr(y_true, y_score)

    return {
        "bpr_loss": np.mean(bpr_losses),
        "auc": auc,
        "average_precision": ap,
        f"ndcg@{topk}": ndcg,
        f"precision@{topk}": prec,
        f"recall@{topk}": rec,
        "mrr": mrr_score
    }
