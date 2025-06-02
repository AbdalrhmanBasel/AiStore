# recommender/gnn/inference.py

import torch
import os
from recommender.utils import load_encoder

# Paths
GNN_EMBEDDINGS_PATH = 'recommender/embeddings/gnn_embeddings.pt'

def get_recommendations(user_id_enc, top_n=5):
    # Load encoders
    user_encoder, product_encoder = load_encoder()
    if user_encoder is None or product_encoder is None:
        print("❌ Encoders not loaded. Cannot generate recommendations.")
        return []

    # Load GNN embeddings
    if not os.path.exists(GNN_EMBEDDINGS_PATH):
        print("❌ GNN embeddings not found. Please run `train_gnn` first.")
        return []

    embeddings = torch.load(GNN_EMBEDDINGS_PATH)

    user_emb = embeddings['user_emb']
    product_emb = embeddings['product_emb']

    if user_id_enc >= user_emb.shape[0]:
        print(f"❌ User index {user_id_enc} out of range. No recommendations.")
        return []

    # Get user embedding
    u_emb = user_emb[user_id_enc].unsqueeze(0)  # shape (1, emb_dim)

    # Compute cosine similarity to all products
    similarities = torch.nn.functional.cosine_similarity(u_emb, product_emb)

    # Get top-N product indices
    top_indices = similarities.argsort(descending=True)[:top_n]

    # Convert to integer list
    top_product_ids_enc = top_indices.cpu().numpy().tolist()

    print(f"✅ get_recommendations() → User {user_id_enc} → top-{top_n}: {top_product_ids_enc}")

    return top_product_ids_enc
