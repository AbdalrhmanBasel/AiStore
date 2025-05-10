from django.http import JsonResponse
from django.apps import apps
import numpy as np

def recommend(request, user_id):
    # Get loaded models and data
    app_config = apps.get_app_config('recommender')
    gnn = app_config.gnn
    predictor = app_config.predictor
    data = app_config.data

    try:
        # Get embeddings
        emb = gnn(data.x_dict, data.edge_index_dict)
        user_emb = emb['user'][user_id].unsqueeze(0)
        item_embs = emb['item']

        # Predict scores
        scores = predictor(user_emb.expand(item_embs.size(0), -1), item_embs).cpu().numpy()

        # Exclude interacted items
        interacted = data['user', 'rates', 'item'].edge_index[1][
            data['user', 'rates', 'item'].edge_index[0] == user_id
        ].cpu().numpy()

        scores[interacted] = -np.inf
        top_k = np.argpartition(scores, -5)[-5:]
        top_k = top_k[np.argsort(-scores[top_k])]

        return JsonResponse({
            "user_id": int(user_id),
            "recommendations": top_k.tolist()
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)