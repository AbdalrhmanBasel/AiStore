import json
import os
from django.conf import settings

# Access the paths defined in settings.py
MAPPING_USER = settings.RECOMMENDER_ROOT / "mappings" / "user_id_map.json"
MAPPING_ITEM = settings.RECOMMENDER_ROOT / "mappings" / "item_id_map.json"
REVERSED_MAPPING_ITEM = settings.RECOMMENDER_ROOT / "mappings" / "id_to_item.json"


def load_mappings():
    # Verify that all the mapping files exist
    for path in [MAPPING_USER, MAPPING_ITEM, REVERSED_MAPPING_ITEM]:
        if not path.exists():
            raise FileNotFoundError(f"Mapping file not found: {path}")
    
    # Load the mappings from the files
    with open(MAPPING_USER, 'r') as f:
        user_id_map = json.load(f)
    
    with open(MAPPING_ITEM, 'r') as f:
        item_id_map = json.load(f)
    
    with open(REVERSED_MAPPING_ITEM, 'r') as f:
        id_to_item = json.load(f)
    
    # Convert keys to correct types
    return {
        'user_id_map': {str(k): int(v) for k, v in user_id_map.items()},
        'item_id_map': {str(k): int(v) for k, v in item_id_map.items()},
        'id_to_item': {int(k): str(v) for k, v in id_to_item.items()}
    }


