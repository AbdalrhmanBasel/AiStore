# recommender/apps.py
from django.apps import AppConfig
import torch
import json
from pathlib import Path
import sys
from django.conf import settings
from .models import HeteroGraphSAGE, LinkPredictor

class RecommenderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recommender'

    gnn = None 
    predictor = None
    full_data = None
    item_id_map = None
    id_to_item = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def ready(self):
        try:
            # Print paths from settings
            print("📊 GNN Model Path:", settings.RECOMMENDER_MODELS['gnn'])
            print("📊 Predictor Model Path:", settings.RECOMMENDER_MODELS['predictor'])
            print("📁 Full Graph Path:", settings.RECOMMENDER_MODELS['graph'])

            # Validate paths
            gnn_path = settings.RECOMMENDER_MODELS['gnn']
            pred_path = settings.RECOMMENDER_MODELS['predictor']
            graph_path = settings.RECOMMENDER_MODELS['graph']

            if not gnn_path.exists():
                raise FileNotFoundError(f"GNN model not found at {gnn_path}")
            if not pred_path.exists():
                raise FileNotFoundError(f"Predictor model not found at {pred_path}")
            if not graph_path.exists():
                raise FileNotFoundError(f"Graph data not found at {graph_path}")

            # Load models with safe unpickling
            torch.serialization.add_safe_globals([HeteroGraphSAGE, LinkPredictor])

            self.gnn = HeteroGraphSAGE(hidden_dim=128).to(self.device)
            self.predictor = LinkPredictor(hidden_dim=128).to(self.device)
            
            # ⚠️ Only load if you trust the source
            self.gnn.load_state_dict(torch.load(
                gnn_path, 
                map_location=self.device,
                weights_only=False
            ))
            self.predictor.load_state_dict(torch.load(
                pred_path, 
                map_location=self.device,
                weights_only=False
            ))
            self.gnn.eval()
            self.predictor.eval()

            # Load graph
            self.full_data = torch.load(graph_path, map_location=self.device, weights_only=False)

            # Load mappings
            with open(settings.RECOMMENDER_MODELS['item_map'], 'r') as f:
                self.item_id_map = json.load(f)
            with open(settings.RECOMMENDER_MODELS['id_to_item'], 'r') as f:
                self.id_to_item = json.load(f)

            print("✅ Recommender system initialized successfully.")
        except Exception as e:
            print(f"🔥 Error initializing recommender: {e}", file=sys.stderr)