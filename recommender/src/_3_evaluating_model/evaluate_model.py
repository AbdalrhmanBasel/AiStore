import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def link_prediction_loss(pos_edge_index, neg_edge_index, z):
    # pos_edge_index: positive edge indices (real edges in the graph)
    # neg_edge_index: negative edge indices (non-existing edges)
    
    # Get node embeddings for positive and negative edges
    pos_edge_emb = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_edge_emb = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    
    # Positive edge score (the higher, the more likely it is a real edge)
    pos_loss = F.logsigmoid(pos_edge_emb).mean()
    
    # Negative edge score (the lower, the more likely it is a fake edge)
    neg_loss = F.logsigmoid(-neg_edge_emb).mean()
    
    return -(pos_loss + neg_loss)




def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, loss_fn: torch.nn.Module, device: str) -> float:
    """
    Evaluate the model on the validation dataset.
    This function calculates the loss over the entire validation set.
    
    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (DataLoader): DataLoader for the validation set.
        loss_fn (torch.nn.Module): The loss function to use for evaluation.
        device (str): The device to run the evaluation on (e.g., "cuda" or "cpu").

    Returns:
        float: The average loss over the entire validation set.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    # Disable gradient calculation for inference
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)  # Move data to device

            # Get node features (x), edge index (edge_index), and labels (y)
            x, edge_index, y = data.x, data.edge_index, data.y

            # Get model predictions
            output = model(x, edge_index)

            # Calculate the loss
            loss = loss_fn(output, y)
            total_loss += loss.item()  # Add loss for this batch
            num_batches += 1

    # Calculate the average loss over all batches
    avg_loss = total_loss / num_batches
    return avg_loss
