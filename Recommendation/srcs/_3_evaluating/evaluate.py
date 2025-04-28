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
    """Evaluate model using torchmetrics retrieval metrics"""
    model.eval()
    
    # Initialize metrics
    precision_metric = RetrievalPrecision(top_k=k)
    recall_metric = RetrievalRecall(top_k=k)
    ndcg_metric = RetrievalNormalizedDCG(top_k=k)
    
    # Move metrics to same device as model
    device = next(model.parameters()).device
    precision_metric = precision_metric.to(device)
    recall_metric = recall_metric.to(device)
    ndcg_metric = ndcg_metric.to(device)
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            targets = torch.cat([torch.ones(batch.edge_index.size[1])], torch.zeros(batch.edge_index.size(1) * 5  )).to(device)
            
            # Get positive scores
            pos_scores = model(batch.x, batch.edge_index)
            
            # Generate negative samples and get their scores
            neg_edges = generate_negative_samples(
                batch.edge_index, 
                batch.num_nodes,
                num_neg_samples=5  # Matches the zeros multiplier above
            )
            neg_scores = model(batch.x, neg_edges)
            
            # Combine and prepare for metrics
            preds = torch.cat([pos_scores, neg_scores])
            indexes = torch.zeros_like(preds, dtype=torch.long)  # Single query
            
            # Update metrics
            precision_metric.update(preds, targets, indexes)
            recall_metric.update(preds, targets, indexes)
            ndcg_metric.update(preds, targets, indexes)
    
    # Compute final metrics
    metrics = {
        'precision@k': precision_metric.compute().item(),
        'recall@k': recall_metric.compute().item(),
        'ndcg@k': ndcg_metric.compute().item()
    }
    
    # Reset metrics for next evaluation
    precision_metric.reset()
    recall_metric.reset()
    ndcg_metric.reset()
    
    return metrics