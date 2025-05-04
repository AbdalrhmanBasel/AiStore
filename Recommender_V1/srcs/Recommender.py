import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from srcs.utils.load_models import load_trained_model


class Recommender:
    def __init__(self, model_path: str, graph_data: Data):
        self.graph = graph_data
        self.model = load_trained_model(model_path)
        self.model.eval()
        self.user_embs, self.item_embs = self.model.get_user_item_embeddings(graph_data)

        # Optional: If you have item metadata
        self.item_metadata = getattr(graph_data, 'item_metadata', None)

    def recommend_top_k_for_user(self, user_id: int, k: int = 10):
        scores = self.user_embs[user_id] @ self.item_embs.T
        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_indices.tolist(), topk_scores.tolist()

    def recommend_similar_items(self, item_id: int, k: int = 10):
        sims = F.cosine_similarity(self.item_embs[item_id].unsqueeze(0), self.item_embs)
        topk_scores, topk_indices = torch.topk(sims, k + 1)
        return [i.item() for i in topk_indices if i != item_id][:k]

    def recommend_similar_users(self, user_id: int, k: int = 10):
        sims = F.cosine_similarity(self.user_embs[user_id].unsqueeze(0), self.user_embs)
        topk_scores, topk_indices = torch.topk(sims, k + 1)
        return [i.item() for i in topk_indices if i != user_id][:k]

    def recommend_users_for_item(self, item_id: int, k: int = 10):
        scores = self.user_embs @ self.item_embs[item_id]
        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_indices.tolist(), topk_scores.tolist()

    def recommend_content_based(self, user_id: int, k: int = 10):
        # Content-based: recommend items similar to those the user interacted with
        if not hasattr(self.graph, 'edge_index'):
            raise ValueError("Graph must have edge_index for content-based recommendation.")

        user_node = user_id
        interacted_items = self.graph.edge_index[1][self.graph.edge_index[0] == user_node].unique()

        if len(interacted_items) == 0:
            return self.recommend_most_popular(k)

        liked_item_embs = self.item_embs[interacted_items]
        avg_profile = liked_item_embs.mean(dim=0)

        scores = F.cosine_similarity(avg_profile.unsqueeze(0), self.item_embs)
        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_indices.tolist(), topk_scores.tolist()

    def recommend_most_popular(self, k: int = 10):
        if not hasattr(self.graph, 'edge_index'):
            raise ValueError("Graph must have edge_index for popularity-based recommendation.")

        _, item_indices = self.graph.edge_index
        item_counts = torch.bincount(item_indices, minlength=self.item_embs.size(0))
        topk = torch.topk(item_counts, k).indices
        return topk.tolist()

    def recommend_recent_items(self, k: int = 10):
        if not hasattr(self.graph, 'item_timestamps'):
            raise ValueError("Graph must have 'item_timestamps' for recent item recommendation.")
        
        timestamps = self.graph.item_timestamps
        _, topk_indices = torch.topk(timestamps, k)
        return topk_indices.tolist()

    def recommend_diverse_items(self, user_id: int, k: int = 10, lambda_div: float = 0.5):
        user_vec = self.user_embs[user_id]
        scores = user_vec @ self.item_embs.T

        selected = []
        candidates = list(range(self.item_embs.size(0)))

        for _ in range(k):
            best_score, best_idx = -float('inf'), -1
            for idx in candidates:
                relevance = scores[idx].item()
                diversity = 0
                if selected:
                    selected_vecs = self.item_embs[selected]
                    diversity = max(F.cosine_similarity(self.item_embs[idx].unsqueeze(0), selected_vecs).tolist())
                mmr = lambda_div * relevance - (1 - lambda_div) * diversity
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            if best_idx == -1:
                break
            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected

    def recommend_for_cold_user(self, k: int = 10):
        # Strategy: fall back to popularity or recent items
        return self.recommend_most_popular(k)

    def recommend_for_all_users(self, k: int = 10):
        scores = self.user_embs @ self.item_embs.T
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        return topk_indices.tolist(), topk_scores.tolist()
