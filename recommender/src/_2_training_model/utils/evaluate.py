import torch 
import torch.nn as nn 

from torch.utils.data import DataLoader

def evaluate_model(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: str) -> float:
    """
    Evaluates the model on a validation set and returns the average loss.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            x, edge_index, y = data.x, data.edge_index, data.y

            # Forward pass
            output = model(x, edge_index)
            loss = loss_fn(output, y)

            total_loss += loss.item() * len(y)
            total_samples += len(y)

    return total_loss / total_samples