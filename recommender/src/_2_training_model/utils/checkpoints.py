import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from torch.optim.lr_scheduler import _LRScheduler
import os

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, 
                    checkpoint_path: str, scheduler: Optional[_LRScheduler] = None) -> None:
    """
    Saves the model, optimizer state, and scheduler state to a checkpoint file.
    """
    checkpoint = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}, loss: {loss:.4f}.")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str, 
                    scheduler: Optional[_LRScheduler] = None) -> Dict[str, Any]:
    """
    Loads a checkpoint (model, optimizer, scheduler states) from a specified path.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from epoch {epoch}, loss: {loss:.4f}.")
    return checkpoint