import os
import torch
from settings import GRAPH_SAVE_PATH, CHECKPOINT_DIR, MODEL_NAME, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_RATE
from src._1_model_selection.GraphSAGEModelV0 import GraphSAGEModelV0


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


def recommender(user_id, features, edge_index, train_edges, top_k=10, model_path="./saved_models/graphsage_model.pt"):
    """
    Recommend items for a given user using the trained model.
    
    Args:
        user_id (int): The ID of the user to recommend items for.
        features (torch.Tensor): Node features.
        edge_index (torch.Tensor): Graph edges.
        train_edges (torch.Tensor): Training edges to exclude seen items.
        top_k (int, optional): The number of top items to recommend (default is 10).
        model_path (str, optional): Path to the saved model checkpoint.
        
    Returns:
        tuple: Top-K item indices and corresponding scores.
    """
    # Load the pretrained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGEModelV0(in_channels=features.size(1), hidden_channels=64, out_channels=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Move data to the appropriate device
    features = features.to(device)
    edge_index = edge_index.to(device)

    # Generate embeddings
    with torch.no_grad():
        z = model(features, edge_index)

    # Compute similarity scores
    if user_id >= len(z):
        raise ValueError(f"User ID {user_id} is out of range.")
    user_embedding = z[user_id]
    scores = torch.matmul(z, user_embedding)

    # Filter seen items
    seen_items = build_seen_item_dict(train_edges).get(user_id, set())
    scores[user_id] = -1e9  # Don't recommend the user to themselves
    for seen_item in seen_items:
        scores[seen_item] = -1e9

    # Get top-K recommendations
    top_k_scores, top_k_indices = torch.topk(scores, top_k)

    return top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()