# recommender/gnn/inference.py — FINAL VERSION 🚀

import torch
import os
from recommender.utils import load_encoder

# --- CONFIG ---
EMBEDDINGS_PATH = 'recommender/embeddings/gnn_embeddings.pt'

# --- Load GNN embeddings ---
def load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        print(f"✅ Loading embeddings from {EMBEDDINGS_PATH}")
        return torch.load(EMBEDDINGS_PATH, map_location=torch.device('cpu'))
    else:
        print("⚠️ No embeddings found!")
        return None

# --- GNN inference: get recommendations ---
def get_recommendations(user_id_enc, top_n=10):
    embeddings = load_embeddings()

    if embeddings is None:
        print("⚠️ No embeddings → returning empty recommendations")
        return []

    try:
        user_emb = embeddings[user_id_enc]

        # Compute similarity to all products
        user_encoder, product_encoder = load_encoder()
        num_users = len(user_encoder.classes_)
        num_products = len(product_encoder.classes_)

        product_emb = embeddings[num_users:num_users + num_products]

        # Cosine similarity
        sim_scores = torch.nn.functional.cosine_similarity(user_emb.unsqueeze(0), product_emb)
        top_indices = torch.topk(sim_scores, top_n).indices.tolist()

        print(f"✅ GNN inference: top {top_n} → {top_indices}")
        return top_indices

    except Exception as e:
        print(f"❌ GNN inference failed: {e}")
        return []
