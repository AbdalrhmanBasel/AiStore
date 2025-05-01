import torch
from typing import List, Tuple
from logger import get_module_logger

logger = get_module_logger("recommend")

@torch.no_grad()
def recommend_top_k(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    user_id: int,
    item_ids: List[int],
    k: int = 10,
    device: str = "cpu"
) -> Tuple[List[int], List[float]]:
    """
    Generate top-k item recommendations for a given user.

    Args:
        model: Trained GNN model.
        x: Node features (num_nodes, num_features).
        edge_index: Graph structure (2, num_edges).
        user_id: ID of the user node in the graph.
        item_ids: List of candidate item node IDs.
        k: Number of top items to recommend.
        device: 'cpu' or 'cuda'.

    Returns:
        Tuple of (recommended item IDs, scores).
    """
    logger.info(f"🔍 Generating top-{k} recommendations for user {user_id}")
    
    model = model.to(device)
    model.eval()

    x = x.to(device)
    edge_index = edge_index.to(device)

    # Construct edge_label_index between user and candidate items
    user_tensor = torch.tensor([user_id] * len(item_ids), device=device)
    item_tensor = torch.tensor(item_ids, device=device)
    edge_label_index = torch.stack([user_tensor, item_tensor], dim=0)

    # Predict scores
    scores = model(x, edge_index, edge_label_index=edge_label_index)
    top_k_scores, top_k_indices = torch.topk(scores, k)

    recommended_items = item_tensor[top_k_indices].cpu().tolist()
    recommended_scores = top_k_scores.cpu().tolist()

    logger.info(f"✅ Top-{k} recommendations: {recommended_items}")
    return recommended_items, recommended_scores
