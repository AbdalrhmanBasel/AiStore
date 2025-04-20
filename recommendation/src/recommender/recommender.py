import torch
from src.models.graphsage import GraphSAGEModel
def build_seen_item_dict(train_edges):
    """
    Build a dictionary of seen items for each user.
    
    Args:
        train_edges (torch.Tensor): Edge indices representing interactions.
        
    Returns:
        dict: Dictionary mapping user IDs to sets of seen items.
    """
    seen_items_by_user = {}
    src_nodes = train_edges[0].tolist()
    dst_nodes = train_edges[1].tolist()
    for u, v in zip(src_nodes, dst_nodes):
        if u not in seen_items_by_user:
            seen_items_by_user[u] = set()
        seen_items_by_user[u].add(v)
    return seen_items_by_user


def recommender(user_id, features, edge_index, train_edges, top_k=10):
    """
    Recommend items for a given user using the trained model.
    
    Args:
        user_id (int): The ID of the user to recommend items for.
        features (torch.Tensor): Node features.
        edge_index (torch.Tensor): Graph edges.
        train_edges (torch.Tensor): Training edges to exclude seen items.
        top_k (int, optional): The number of top items to recommend (default is 10).
        
    Returns:
        tuple: Top-K item indices and corresponding scores.
    """
    model = GraphSAGEModel(in_channels=features.size(1), hidden_channels=64, out_channels=32)
    model.load_state_dict(torch.load("./saved_models/graphsage_model.pt"))
    model.eval()

    with torch.no_grad():
        z = model(features, edge_index)

    user_embedding = z[user_id]
    scores = torch.matmul(z, user_embedding)

    seen_items = build_seen_item_dict(train_edges).get(user_id, set())
    scores[user_id] = -1e9  # Don't recommend the user to themselves

    for seen_item in seen_items:
        scores[seen_item] = -1e9

    top_k_scores, top_k_indices = torch.topk(scores, top_k)

    return top_k_indices, top_k_scores
