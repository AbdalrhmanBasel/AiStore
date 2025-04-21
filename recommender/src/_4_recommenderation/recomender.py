import torch
import os
from settings import GRAPH_SAVE_PATH, CHECKPOINT_DIR, MODEL_NAME

def generate_recommendations(user_id, top_k=5):
    """
    Generate recommendations for a specific user using the trained GNN model.

    Args:
        user_id (int): The ID of the user for whom recommendations are generated.
        top_k (int): The number of top recommendations to return.

    Returns:
        list: A list of recommended item IDs for the user.
    """
    print(f"Generating recommendations for user {user_id}...")

    # Step 1: Load the trained model
    model_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    # Step 2: Load the graph data
    graph_data_path = GRAPH_SAVE_PATH
    if not os.path.exists(graph_data_path):
        raise FileNotFoundError(f"Graph data not found at {graph_data_path}")
    
    graph_data = torch.load(graph_data_path)
    graph_data.to(device)

    # Step 3: Compute node embeddings using the trained model
    with torch.no_grad():
        node_embeddings = model(graph_data.x, graph_data.edge_index)

    # Step 4: Extract user and item embeddings
    user_embedding = node_embeddings[user_id].unsqueeze(0)  # Shape: [1, embedding_dim]
    item_embeddings = node_embeddings[graph_data.num_users:]  # Items start after user nodes

    # Step 5: Compute similarity scores (e.g., cosine similarity)
    similarity_scores = torch.nn.functional.cosine_similarity(user_embedding, item_embeddings)

    # Step 6: Get top-k recommendations
    top_k_scores, top_k_indices = torch.topk(similarity_scores, top_k)
    recommended_item_ids = top_k_indices.cpu().numpy().tolist()

    print(f"Top-{top_k} recommendations for user {user_id}: {recommended_item_ids}")

    return recommended_item_ids