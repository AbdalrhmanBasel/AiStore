from logger import get_module_logger
import torch 
from srcs._3_evaluating.losses.bce_loss import generate_negative_samples
from torchmetrics.retrieval import RetrievalPrecision, RetrievalRecall, RetrievalNormalizedDCG

import os
import sys

from logger import get_module_logger

logger = get_module_logger("evaluater")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)



from logger import get_module_logger
import torch
from torch import Tensor
from typing import Dict
from torch_geometric.data import DataLoader
from torchmetrics.retrieval import (
    RetrievalPrecision,
    RetrievalRecall,
    RetrievalNormalizedDCG
)
from srcs._3_evaluating.losses.bce_loss import generate_negative_samples

logger = get_module_logger("evaluater")

def evaluater():
    logger.info("🔄 Starting evaluating process.")
    logger.info("✅ Evaluating process completed.")


def evaluate(model, val_loader, k=10) -> Dict[str, float]:
    model.eval()

    # Define the device based on where the model is located (GPU or CPU)
    device = next(model.parameters()).device

    # Initialize metrics and move them to the device
    precision_metric = RetrievalPrecision(top_k=k).to(device)
    recall_metric    = RetrievalRecall(top_k=k).to(device)
    ndcg_metric      = RetrievalNormalizedDCG(top_k=k).to(device)

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # 1) Grab the POSITIVE edges for this batch:
            pos_edge_index = batch.edge_label_index           # shape [2, num_pos]
            num_pos = pos_edge_index.size(1)

            # 2) Build targets = [1,...,1, 0,...,0] of length (num_pos + num_neg)
            num_neg = num_pos * 5
            targets = torch.cat((
                torch.ones(num_pos, device=device),
                torch.zeros(num_neg, device=device)
            ))

            # 3) Get model scores on positives
            pos_scores = model(batch.x, pos_edge_index)       # shape [num_pos]

            # 4) Generate and score NEGATIVE edges
            neg_edge_index = generate_negative_samples(
                pos_edge_index, 
                batch.num_nodes,
                num_neg_samples=5
            )
            neg_scores = model(batch.x, neg_edge_index)       # shape [num_neg]

            # 5) Concatenate predictions and build `indexes`
            preds   = torch.cat((pos_scores, neg_scores))     # shape [num_pos+num_neg]
            indexes = torch.zeros_like(preds, dtype=torch.long)

            # 6) Update metrics
            precision_metric.update(preds, targets, indexes)
            recall_metric.update(preds, targets, indexes)
            ndcg_metric.update(preds, targets, indexes)

    # Compute final metrics
    metrics = {
        'precision@k': precision_metric.compute().item(),
        'recall@k':    recall_metric.compute().item(),
        'ndcg@k':      ndcg_metric.compute().item()
    }

    # Reset metrics for next evaluation
    precision_metric.reset()
    recall_metric.reset()
    ndcg_metric.reset()

    return metrics
