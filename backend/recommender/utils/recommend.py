import numpy as np
import pandas as pd

from django.apps import apps


def get_recommendations(user_id, k=5):
    """
    Generate top-k recommendations for a user.
    """
    app_config = apps.get_app_config('recommender')
    
    if not app_config.gnn or not app_config.full_data:
        raise ValueError("Recommender not initialized")
    
    if str(user_id) not in app_config.item_id_map:
        raise ValueError(f"User '{user_id}' not found in mappings")

    user_idx = int(app_config.item_id_map[str(user_id)])
    emb = app_config.gnn(app_config.full_data.x_dict, app_config.full_data.edge_index_dict)

    # Predict scores
    user_emb = emb['user'][user_idx].unsqueeze(0)
    item_embs = emb['item']
    scores = app_config.predictor(
        user_emb.expand(item_embs.size(0), -1),
        item_embs
    ).cpu().numpy()

    # Exclude interacted items
    edge_index = app_config.full_data['user', 'rates', 'item'].edge_index
    interacted_items = edge_index[1][edge_index[0] == user_idx].cpu().numpy()
    scores[interacted_items] = -np.inf

    # Get top-k indices
    top_k_indices = np.argpartition(scores, -k)[-k:]
    return [app_config.id_to_item.get(str(int(i)), None) for i in top_k_indices]


def get_recommended_products(recommended, meta_df_clean):
    """
    Safely filter and reorder meta_df_clean based on recommended ASINs.
    
    Args:
        recommended (list): List of ASINs from recommendation system
        meta_df_clean (pd.DataFrame): Cleaned metadata with item features
    
    Returns:
        pd.DataFrame: Filtered and ordered DataFrame with product details
    """
    # Step 1: Filter only valid ASINs that exist in meta_df_clean
    valid_recommended = [asin for asin in recommended if asin in meta_df_clean['parent_asin'].values]
    
    # Step 2: Filter meta_df_clean for matching ASINs
    filtered = meta_df_clean[meta_df_clean['parent_asin'].isin(valid_recommended)]
    
    # Step 3: Reorder to match recommendation order
    filtered['asin_order'] = pd.Categorical(
        filtered['parent_asin'], 
        categories=valid_recommended,
        ordered=True
    )
    return filtered.sort_values('asin_order').drop('asin_order', axis=1)
