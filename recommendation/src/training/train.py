import torch
import logging
from torch.optim import Optimizer
from torch_geometric.data import Data

from src.training.callbacks import get_callbacks
from src.training.eval import evaluate
from src.training.loss import compute_loss
from src.utils.link_utils import decode_link  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, data, optimizer, train_pos, train_neg):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model to train.
        data (torch_geometric.data.Data): The input graph data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        train_pos (torch.Tensor): Positive training edge indices.
        train_neg (torch.Tensor): Negative training edge indices.
        
    Returns:
        float: The training loss.
    """
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)

    pos_out = decode_link(z, train_pos)
    neg_out = decode_link(z, train_neg)
    
    loss = compute_loss(pos_out, neg_out)
    loss.backward()
    optimizer.step()
    return loss.item()