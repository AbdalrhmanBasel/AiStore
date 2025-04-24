# train_step.py
import torch
import torch.nn as nn
from typing import Optional, Any


def train_step(model: nn.Module, data: Any, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
               gradient_clip: Optional[float] = None, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> torch.Tensor:
    model.train()

    # Extract node features, edge index, and labels from the data
    x, edge_index, y = data.x, data.edge_index, data.y

    # Safety check for labels
    if y is None:
        raise ValueError("Labels (y) are missing in the data. Check preprocessing.")

    # Zero gradients
    optimizer.zero_grad()

    # Mixed precision training
    with torch.cuda.amp.autocast(enabled=True):  # Use autocast for mixed precision
        # Forward pass: get embeddings
        output = model(x, edge_index)

        # Compute edge scores (dot product between connected nodes)
        src_embeddings = output[edge_index[0]]  # Source node embeddings
        dst_embeddings = output[edge_index[1]]  # Destination node embeddings
        edge_scores = (src_embeddings * dst_embeddings).sum(dim=1)  # Dot product for edge scores

        # Compute loss
        loss = loss_fn(edge_scores, y.float())  # Ensure labels are floats

    # Backward pass
    if scaler:
        scaler.scale(loss).backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

    return loss