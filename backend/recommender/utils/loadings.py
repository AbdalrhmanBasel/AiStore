import numpy as np
import pandas as pd
from django.conf import settings
from django.apps import apps  
import json



def load_mappings():
    """
    Load user/item mappings and ASIN → item index
    """
    # Load mappings from JSON files
    with open(settings.MAPPING_USER_T0_ID_PATH, "r") as f:
        user_id_map = json.load(f)
    with open(settings.MAPPING_ITEM_ASIN_TO_ID_PATH, "r") as f:
        item_id_map = json.load(f)
    with open(settings.MAPPING_ID_TO_ITEM_PATH, "r") as f:
        id_to_item = json.load(f)

    return {
        'user_id_map': {str(k): int(v) for k, v in user_id_map.items()},
        'item_id_map': {str(k): int(v) for k, v in item_id_map.items()},
        'id_to_item': {int(k): str(v) for k, v in id_to_item.items()}
    }

def load_graph_and_models():
    """
    Get preloaded models and graph from RecommenderConfig
    """
    try:
        # Get app config from Django registry
        app_config = apps.get_app_config('recommender')
        
        # Validate required attributes
        if not all([app_config.gnn, app_config.predictor, app_config.full_data]):
            raise ValueError("Recommender models not initialized")
            
        return (
            app_config.full_data,
            app_config.gnn,
            app_config.predictor,
            app_config.item_id_map,
            app_config.id_to_item
        )
    except Exception as e:
        print(f"🔥 Error loading recommender: {e}", file=sys.stderr)
        raise