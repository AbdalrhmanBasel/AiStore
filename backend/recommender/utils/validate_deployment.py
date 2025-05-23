# validate_deployment.py
import torch
import numpy as np
from django.apps import apps

def validate_recommender():
    try:
        app_config = apps.get_app_config('recommender')
        print("✅ RecommenderConfig loaded")

        # Test model dimensions
        print("📊 User feature shape:", app_config.full_data['user'].x.shape)
        print("📊 Item feature shape:", app_config.full_data['item'].x.shape)
        print("📊 Edge index shape:", app_config.full_data['user', 'rates', 'item'].edge_index.shape)

        # Test prediction
        user_id = "AGBFYI2DDIKXC5Y4FARTYDTQBMFQ"
        if str(user_id) in app_config.item_id_map:
            user_idx = int(app_config.item_id_map[str(user_id)])
            emb = app_config.gnn(app_config.full_data.x_dict, app_config.full_data.edge_index_dict)

            scores = app_config.predictor(
                emb['user'][user_idx].unsqueeze(0).expand(emb['item'].size(0), -1),
                emb['item']
            ).cpu().numpy()

            top_k = np.argpartition(scores, -5)[-5:]
            print("✅ Sample predictions:", [app_config.id_to_item.get(str(i)) for i in top_k])
        else:
            print("⚠️ User not found in mappings - fallback to popular products")
    except Exception as e:
        print("❌ Validation failed:", str(e))