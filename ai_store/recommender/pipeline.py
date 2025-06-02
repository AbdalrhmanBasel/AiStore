# recommender/pipeline.py

from recommender.graph.graph_export import export_interaction_graph
from recommender.graph.loader import build_graph_data
from recommender.gnn.train import train_gnn
from recommender.models import Recommendation
from recommender.utils import get_products_from_encoded, user_encoder
import torch

EMBEDDINGS_PATH = 'recommender/embeddings/gnn_embeddings.pt'

def run_full_pipeline():
    print("🚀 Step 1/4: Exporting interaction graph...")
    export_interaction_graph()

    print("🚀 Step 2/4: Building graph data...")
    build_graph_data()

    print("🚀 Step 3/4: Training GNN...")
    train_gnn()

    print("🚀 Step 4/4: Updating recommendations...")
    update_recommendations()

    print("🎉 Full pipeline completed!")

def update_recommendations(top_n=10):
    # Load embeddings
    emb = torch.load(EMBEDDINGS_PATH)
    user_emb = emb['user_emb']
    product_emb = emb['product_emb']

    num_users = user_emb.shape[0]

    # Clear old recommendations
    Recommendation.objects.all().delete()

    for user_enc in range(num_users):
        user_vector = user_emb[user_enc].unsqueeze(0)
        scores = torch.nn.functional.cosine_similarity(user_vector, product_emb)
        top_indices = scores.argsort(descending=True)[:top_n]
        top_scores = scores[top_indices]

        product_ids_enc = top_indices.tolist()
        products = get_products_from_encoded(product_ids_enc)

        user_id = reverse_user_encoder(user_enc)

        for i, product in enumerate(products):
            if product:
                Recommendation.objects.create(
                    user_id=user_id,
                    product=product,
                    score=float(top_scores[i].item())
                )

    print("✅ Recommendations updated")

def reverse_user_encoder(user_enc_id):
    return int(user_encoder.inverse_transform([user_enc_id])[0])
