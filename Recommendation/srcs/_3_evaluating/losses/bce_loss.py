import torch
import torch.nn.functional as F


import os
import sys

from logger import get_module_logger

logger = get_module_logger("losses")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)

def bce_loss(pos_score, neg_score):
    pos_loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))
    neg_loss = F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
    return pos_loss + neg_loss

def generate_negative_samples(edge_index, num_nodes, num_neg_samples=5):
    neg_edges = torch.randint(0, num_nodes, (2, edge_index.size(1) * num_neg_samples))
    return neg_edges

