import torch 
import torch.nn as nn


from logger import get_module_logger

from srcs._1_modeling.GraphSAGE import GraphSAGE

import os
import sys


logger = get_module_logger("Recommendation Model")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)


class RecommendationModel(nn.Module):
    """
    Recommendation model using dual GraphSAGE encoders (for users and items) and a scoring head.
    """

    def __init__(self, user_dim: int, item_dim: int, hidden_dim: int = 256, embed_dim: int = 64, dropout: float = 0.5):
        super(RecommendationModel, self).__init__()

        self.user_encoder = GraphSAGE(user_dim, hidden_dim, embed_dim, dropout=dropout)
        self.item_encoder = GraphSAGE(item_dim, hidden_dim, embed_dim, dropout=dropout)

        self.scoring_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        logger.info(f"Initialized RecommendationModel | "
                    f"User_dim: {user_dim}, Item_dim: {item_dim}, Hidden: {hidden_dim}, Embed: {embed_dim}")

    def forward(self, user_x: torch.Tensor, item_x: torch.Tensor,
                user_edge_index: torch.Tensor, item_edge_index: torch.Tensor) -> torch.Tensor:
        logger.debug("Starting forward pass of RecommendationModel...")

        user_emb = self.user_encoder(user_x, user_edge_index)
        item_emb = self.item_encoder(item_x, item_edge_index)

        combined = torch.cat([user_emb, item_emb], dim=-1)
        scores = torch.sigmoid(self.scoring_head(combined)).squeeze()

        logger.debug("Forward pass of RecommendationModel complete.")
        return scores