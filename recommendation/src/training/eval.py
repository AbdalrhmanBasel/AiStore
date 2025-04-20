import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.utils.link_utils import decode_link  

def evaluate_while_training(model, data, test_pos, test_neg):
    """
    Evaluate the model on test data.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        data (torch_geometric.data.Data): The input graph data.
        test_pos (torch.Tensor): Positive test edge indices.
        test_neg (torch.Tensor): Negative test edge indices.
        
    Returns:
        tuple: Accuracy, AUC, precision, recall, F1 score, and node embeddings.
    """
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        pos_out = decode_link(z, test_pos)
        neg_out = decode_link(z, test_neg)

        pos_pred = torch.sigmoid(pos_out)
        neg_pred = torch.sigmoid(neg_out)

        # True labels for evaluation
        y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))])
        y_pred = torch.cat([pos_pred, neg_pred])

        # Compute AUC
        auc = roc_auc_score(y_true.cpu(), y_pred.cpu())

        # Compute Precision, Recall, and F1-Score
        y_pred_bin = (y_pred > 0.5).float()
        precision = precision_score(y_true.cpu(), y_pred_bin.cpu())
        recall = recall_score(y_true.cpu(), y_pred_bin.cpu())
        f1 = f1_score(y_true.cpu(), y_pred_bin.cpu())

        # Accuracy
        correct = (y_pred_bin == y_true).sum().item()
        accuracy = correct / y_true.size(0)

        return accuracy, auc, precision, recall, f1, z


def evaluate_link_prediction(z, pos_edge_index, neg_edge_index, k=10):
    """
    Evaluate link prediction using Mean Reciprocal Rank (MRR) and Hit@K metrics.
    
    Args:
        z (torch.Tensor): Node embeddings.
        pos_edge_index (torch.Tensor): Positive test edge indices.
        neg_edge_index (torch.Tensor): Negative test edge indices.
        k (int, optional): The top-K rank threshold for Hit@K (default is 10).
        
    Returns:
        tuple: MRR and Hit@K scores.
    """
    src_pos, dst_pos = pos_edge_index
    src_neg, dst_neg = neg_edge_index

    pos_scores = (z[src_pos] * z[dst_pos]).sum(dim=1)
    neg_scores = (z[src_neg] * z[dst_neg]).sum(dim=1)

    scores = torch.cat([pos_scores.unsqueeze(1), neg_scores.unsqueeze(1)], dim=1)
    
    mrr_total = 0.0
    hits_total = 0.0
    num_edges = pos_scores.size(0)

    for i in range(num_edges):
        pos_score = scores[i, 0]
        all_scores = torch.cat([pos_score.view(1), scores[i, 1:]])
        rank = (all_scores > pos_score).sum() + 1  # 1-based rank
        mrr_total += 1.0 / rank
        if rank <= k:
            hits_total += 1.0

    mrr = mrr_total / num_edges
    hits_at_k = hits_total / num_edges

    return mrr, hits_at_k




import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.utils.link_utils import decode_link  
from src.models.graphsage import GraphSAGEModel

def evaluate(model, data, test_pos, test_neg):
    """
    Evaluate the model on test data.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        data (torch_geometric.data.Data): The input graph data.
        test_pos (torch.Tensor): Positive test edge indices.
        test_neg (torch.Tensor): Negative test edge indices.
        
    Returns:
        tuple: Accuracy, AUC, precision, recall, F1 score, and node embeddings.
    """
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        pos_out = decode_link(z, test_pos)
        neg_out = decode_link(z, test_neg)

        pos_pred = torch.sigmoid(pos_out)
        neg_pred = torch.sigmoid(neg_out)

        # True labels for evaluation
        y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))])
        y_pred = torch.cat([pos_pred, neg_pred])

        # Compute AUC
        auc = roc_auc_score(y_true.cpu(), y_pred.cpu())

        # Compute Precision, Recall, and F1-Score
        y_pred_bin = (y_pred > 0.5).float()
        precision = precision_score(y_true.cpu(), y_pred_bin.cpu())
        recall = recall_score(y_true.cpu(), y_pred_bin.cpu())
        f1 = f1_score(y_true.cpu(), y_pred_bin.cpu())

        # Accuracy
        correct = (y_pred_bin == y_true).sum().item()
        accuracy = correct / y_true.size(0)

        return accuracy, auc, precision, recall, f1, z


def evaluate_link_prediction(z, pos_edge_index, neg_edge_index, k=10):
    """
    Evaluate link prediction using Mean Reciprocal Rank (MRR) and Hit@K metrics.
    
    Args:
        z (torch.Tensor): Node embeddings.
        pos_edge_index (torch.Tensor): Positive test edge indices.
        neg_edge_index (torch.Tensor): Negative test edge indices.
        k (int, optional): The top-K rank threshold for Hit@K (default is 10).
        
    Returns:
        tuple: MRR and Hit@K scores.
    """
    src_pos, dst_pos = pos_edge_index
    src_neg, dst_neg = neg_edge_index

    pos_scores = (z[src_pos] * z[dst_pos]).sum(dim=1)
    neg_scores = (z[src_neg] * z[dst_neg]).sum(dim=1)

    scores = torch.cat([pos_scores.unsqueeze(1), neg_scores.unsqueeze(1)], dim=1)
    
    mrr_total = 0.0
    hits_total = 0.0
    num_edges = pos_scores.size(0)

    for i in range(num_edges):
        pos_score = scores[i, 0]
        all_scores = torch.cat([pos_score.view(1), scores[i, 1:]])
        rank = (all_scores > pos_score).sum() + 1  # 1-based rank
        mrr_total += 1.0 / rank
        if rank <= k:
            hits_total += 1.0

    mrr = mrr_total / num_edges
    hits_at_k = hits_total / num_edges

    return mrr, hits_at_k






def compute_classification_metrics(pos_scores: torch.Tensor, neg_scores: torch.Tensor):
    """Compute binary classification metrics from link prediction scores."""
    y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    y_pred = torch.cat([torch.sigmoid(pos_scores), torch.sigmoid(neg_scores)])
    y_pred_bin = (y_pred > 0.5).float()

    accuracy = (y_pred_bin == y_true).float().mean().item()
    auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
    precision = precision_score(y_true.cpu(), y_pred_bin.cpu())
    recall = recall_score(y_true.cpu(), y_pred_bin.cpu())
    f1 = f1_score(y_true.cpu(), y_pred_bin.cpu())

    return accuracy, auc, precision, recall, f1


def evaluate_classification_metrics(z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor):
    """Decode links and compute classification metrics."""
    pos_scores = decode_link(z, pos_edge_index)
    neg_scores = decode_link(z, neg_edge_index)
    return compute_classification_metrics(pos_scores, neg_scores)


def compute_ranking_metrics(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int = 10):
    """Compute MRR and Hits@K from raw scores."""
    mrr_total = 0.0
    hits_total = 0.0
    num_edges = pos_scores.size(0)

    for i in range(num_edges):
        pos_score = pos_scores[i]
        all_scores = torch.cat([pos_score.view(1), neg_scores[i]])
        rank = (all_scores > pos_score).sum().item() + 1  # 1-based rank

        mrr_total += 1.0 / rank
        if rank <= k:
            hits_total += 1.0

    mrr = mrr_total / num_edges
    hits_at_k = hits_total / num_edges
    return mrr, hits_at_k


def evaluate_ranking_metrics(z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor, k: int = 10):
    """Compute ranking metrics (MRR, Hits@K) for link prediction."""
    src_pos, dst_pos = pos_edge_index
    src_neg, dst_neg = neg_edge_index

    pos_scores = (z[src_pos] * z[dst_pos]).sum(dim=1)
    neg_scores = (z[src_neg] * z[dst_neg]).sum(dim=1)

    # For each positive edge, associate one or more negative samples
    neg_scores_per_pos = neg_scores.view(-1, 1)  # You may adapt this depending on your sampling strategy
    return compute_ranking_metrics(pos_scores, neg_scores_per_pos, k)


def evaluate(model_path: str, embed_path: str, data, test_pos_edge, test_neg_edge, k: int = 10):
    """
    Full evaluation pipeline: classification + ranking.

    Args:
        model_path (str): Path to trained model.
        embed_path (str): Optional path to precomputed embeddings.
        data (torch_geometric.data.Data): Full graph data.
        test_pos_edge (Tensor): Positive test edges.
        test_neg_edge (Tensor): Negative test edges.
        k (int): Hits@K threshold.

    Returns:
        dict: Dictionary of all evaluation metrics.
    """
    model = GraphSAGEModel(in_channels=data.x.size(1), hidden_channels=64, out_channels=32)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if embed_path:
        z = torch.load(embed_path)
    else:
        with torch.no_grad():
            z = model(data.x, data.edge_index)

    accuracy, auc, precision, recall, f1 = evaluate_classification_metrics(z, test_pos_edge, test_neg_edge)
    mrr, hits_at_k = evaluate_ranking_metrics(z, test_pos_edge, test_neg_edge, k)

    print("\n📊 Model Evaluation Metrics:")
    print(f"✅ Accuracy:  {accuracy:.4f}")
    print(f"✅ AUC:       {auc:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"⭐ MRR:       {mrr:.4f}")
    print(f"⭐ Hits@{k}:  {hits_at_k:.4f}")

    return {
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "hits@k": hits_at_k
    }
