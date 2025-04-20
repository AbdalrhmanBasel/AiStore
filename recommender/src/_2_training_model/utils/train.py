import recommender.settings as cfg
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from utils.checkpoints import save_checkpoint, load_checkpoint
from utils.evaluate import evaluate_model
from typing import Optional, Dict, Any
from device import logger
from torch.cuda.amp import autocast, GradScaler
from utils.device import get_best_device


def train_graphsage_model(
        model: nn.Module, optimizer: torch.optim.Optimizer,
        early_stopping: Optional[Dict[str, Any]] = None,
        epochs: int = cfg.EPOCHS,
        resume_from_checkpoint: bool = False, 
        checkpoint_path: Optional[str] = None,
        graph_loader: Optional[DataLoader] = None,  
        val_dataloader: Optional[DataLoader] = None,
        train_step: Optional[Any] = None, 
        loss_fn: Optional[nn.Module] = None, 
        gradient_clip: Optional[float] = None,  
        print_interval: int = 10  
    ):
    """
    Trains the GraphSAGE model.

    Args:
        model (nn.Module): The GraphSAGE model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for model training.
        early_stopping (Optional[Dict[str, Any]]): Early stopping configuration (metric, patience).
        epochs (int): Total number of training epochs.
        resume_from_checkpoint (bool): Whether to resume training from a checkpoint.
        checkpoint_path (Optional[str]): Path to load and save checkpoints.
        graph_loader (Optional[DataLoader]): DataLoader for training data.
        val_dataloader (Optional[DataLoader]): DataLoader for validation data.
        train_step (Optional[Any]): The training step function to be used.
        loss_fn (Optional[nn.Module]): Loss function for training.
        gradient_clip (Optional[float]): Maximum value for gradient clipping.
        print_interval (int): Interval to log training progress.

    Returns:
        None
    """
    device = get_best_device()
    model.to(device)

    scaler = GradScaler()

    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if early_stopping:
        best_metric = float("inf") if early_stopping["metric"] == "loss" else -float("inf")
        patience = early_stopping["patience"]
        epochs_without_improvement = 0

    start_epoch = 0
    if resume_from_checkpoint and checkpoint_path:
        checkpoint = load_checkpoint(model, optimizer, checkpoint_path, scheduler)
        start_epoch = checkpoint['epoch']
    else:
        checkpoint = {}

    tb_writer = SummaryWriter()

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        # Iterate over mini-batches sampled by NeighborLoader
        for batch in graph_loader:
            batch = batch.to(device) 

            loss = train_step(model, batch, loss_fn, optimizer, gradient_clip, scaler)
            epoch_loss += loss.item()

        epoch_loss /= len(graph_loader)
        tb_writer.add_scalar('Loss/train', epoch_loss, epoch)

        if val_dataloader:
            val_loss = evaluate_model(model, val_dataloader, loss_fn, device)
            tb_writer.add_scalar('Loss/val', val_loss, epoch)

        scheduler.step()

        if (epoch + 1) % print_interval == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
            if val_dataloader:
                logger.info(f"Validation Loss: {val_loss:.4f}")

        if early_stopping:
            metric = epoch_loss if early_stopping["metric"] == "loss" else 0
            if metric < best_metric:
                best_metric = metric
                epochs_without_improvement = 0
                
                if checkpoint_path:
                    save_checkpoint(model, optimizer, epoch + 1, epoch_loss, checkpoint_path, scheduler)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} due to no improvement in {early_stopping['metric']}")
                    break

    logger.info("Training completed.")

    if checkpoint_path:
        save_checkpoint(model, optimizer, epochs, epoch_loss, checkpoint_path, scheduler)


def train_step(model: nn.Module, data: Any, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, 
               gradient_clip: Optional[float] = None, scaler: Optional[GradScaler] = None) -> torch.Tensor:
    """
    Performs a single training step with mixed precision support (autocast) for efficiency.

    Args:
        model (nn.Module): The model to be trained.
        data (Any): The input data (includes features and labels).
        loss_fn (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer used for the step.
        gradient_clip (Optional[float]): Maximum value for gradient clipping.
        scaler (Optional[GradScaler]): A scaler for mixed precision training.

    Returns:
        torch.Tensor: The computed loss value for the current step.
    """
    model.train()
    
    x, edge_index, y = data.x, data.edge_index, data.y
    optimizer.zero_grad()

    with autocast(enabled=True): 
        output = model(x, edge_index)
        loss = loss_fn(output, y)

    if scaler: 
        scaler.scale(loss).backward()
    else:
        loss.backward()

    if gradient_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return loss
