from django.apps import AppConfig
import torch
from .models import HeteroGraphSAGE, LinkPredictor
from pathlib import Path
import os
class RecommenderConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "recommender"

    # Load models and data at startup
    gnn = None
    predictor = None
    data = None

    def ready(self):
        MODEL_DIR = Path(__file__).parent / "utils"

        # Load GNN model
        self.gnn = HeteroGraphSAGE(hidden_dim=128).to("cpu")
        self.gnn.load_state_dict(torch.load(MODEL_DIR / "gnn_model.pth", map_location="cpu"))
        self.gnn.eval()

        # Load predictor
        self.predictor = LinkPredictor(hidden_dim=128).to("cpu")
        self.predictor.load_state_dict(torch.load(MODEL_DIR / "predictor_model.pth", map_location="cpu"))
        self.predictor.eval()

        # Load preprocessed HeteroData
        self.data = torch.load(MODEL_DIR / "data.pt")