# srcs/_4_recommending/recommender.py

import os
import sys
import torch
from logger import get_module_logger

from srcs._1_modeling.GraphSAGE import GraphSAGE
from srcs._2_training.utils.load_model import load_model, load_graph_data
from settings import TRAINED_MODEL_PATH, DEVICE

logger = get_module_logger("Recommender")

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def load_model_and_graph():
    # 1) Instantiate & load the trained GraphSAGE
    model = GraphSAGE().to(DEVICE)
    model, _, _ = load_model(
        model=model,
        optimizer=None,
        checkpoint_path=TRAINED_MODEL_PATH,
        device=DEVICE
    )
    model.eval()

    # 2) Load the graph Data (features + edge_index)
    graph = load_graph_data()
    x = graph.x.to(DEVICE)
    edge_index = graph.edge_index.to(DEVICE)

    logger.info("✅ Model and graph data ready.")
    return model, x, edge_index


def recommend_top_k(model, x, edge_index, user_id, k=10):
    """
    Compute all node embeddings, then score each node by dot(u,v)
    where u is the user embedding and v is the candidate embedding.
    Return the top-k item indices & scores.
    """
    with torch.no_grad():
        z = model(x, edge_index)              # [num_nodes, out_dim]
        u = z[user_id]                        # [out_dim]
        scores = (z @ u).cpu()                # [num_nodes]

    topk_scores, topk_idx = torch.topk(scores, k)
    return topk_idx.tolist(), topk_scores.tolist()


def log_recs(items, scores, user_id):
    logger.info(f"🔝 Top-{len(items)} recs for user {user_id}:")
    for i, (it, sc) in enumerate(zip(items, scores), 1):
        logger.info(f"-----> {i}. Node {it} — score {sc:.4f}")


def recommender(user_id: int = 42, top_k: int = 10):
    """
    Full pipeline: load model+graph, run top-k, log & return.
    Returns:
        List[int]: Top-k recommended node IDs.
    """
    model, x, edge_index = load_model_and_graph()

    if not (0 <= user_id < x.size(0)):
        raise ValueError(f"User ID must be in [0, {x.size(0)-1}]")

    items, scores = recommend_top_k(model, x, edge_index, user_id, k=top_k)
    log_recs(items, scores, user_id)

    # only return the item IDs
    return items