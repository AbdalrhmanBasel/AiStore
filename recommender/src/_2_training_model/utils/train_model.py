# train_model.py
import os
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any
from src._2_training_model.utils.checkpoints import save_checkpoint, load_checkpoint
from src._2_training_model.train_step import train_step
from recommender.src._3_evaluating_model.evaluate_model import evaluate_model
from settings import CHECKPOINT_DIR, TENSORBOARD_LOG_DIR

def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader],
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr: float = 1e-3,
    epochs: int = 50,
    device: str = "cpu",
    scheduler: Optional[_LRScheduler] = None,
    gradient_clip: Optional[float] = None,
    early_stopping: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[str] = None,
    resume_from_checkpoint: bool = False,
) -> None:
    """
    A customizable training loop with mixed precision support, early stopping, and TensorBoard logging.
    
    Args:
        model (torch.nn.Module): The neural network model being trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_dataloader (Optional[torch.utils.data.DataLoader]): DataLoader for the validation dataset.
        loss_fn (torch.nn.Module): The loss function used to compute the training loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        device (str): Device to run training on ("cpu" or "cuda").
        scheduler (Optional[_LRScheduler]): Learning rate scheduler for adaptive learning rates.
        gradient_clip (Optional[float]): Maximum norm for gradient clipping.
        early_stopping (Optional[Dict[str, Any]]): Configuration for early stopping (e.g., patience, metric).
        checkpoint_path (Optional[str]): Path to save/load model checkpoints.
        resume_from_checkpoint (bool): Whether to resume training from a saved checkpoint.
    """
    # Set device and move model to the correct device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize mixed-precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Initialize or load the scheduler
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize early stopping
    if early_stopping:
        best_metric = float("inf") if early_stopping["metric"] == "loss" else -float("inf")
        patience = early_stopping["patience"]
        epochs_without_improvement = 0

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_from_checkpoint and checkpoint_path:
        checkpoint = load_checkpoint(model, optimizer, checkpoint_path, scheduler)
        start_epoch = checkpoint['epoch']
    else:
        checkpoint = {}

    # TensorBoard setup
    tb_writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        # Iterate over training data
        for data in train_dataloader:
            data = data.to(device)  # Move data to device
            loss = train_step(model, data, loss_fn, optimizer, gradient_clip, scaler)
            epoch_loss += loss.item()

        # Average loss over the epoch
        epoch_loss /= len(train_dataloader)

        # Logging to TensorBoard
        tb_writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Validation
        if val_dataloader:
            val_loss = evaluate_model(model, val_dataloader, loss_fn, device)
            tb_writer.add_scalar('Loss/val', val_loss, epoch)

        # Learning rate scheduler step
        scheduler.step()

        # Logging progress
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        if val_dataloader:
            print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if early_stopping:
            metric = epoch_loss if early_stopping["metric"] == "loss" else 0
            if metric < best_metric:
                best_metric = metric
                epochs_without_improvement = 0
                # Save checkpoint
                if checkpoint_path:
                    save_checkpoint(model, optimizer, epoch + 1, epoch_loss, checkpoint_path, scheduler)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1} due to no improvement in {early_stopping['metric']}")
                    break

    print("Training completed.")

    # Final model save
    if checkpoint_path:
        save_checkpoint(model, optimizer, epochs, epoch_loss, checkpoint_path, scheduler)

    # Close TensorBoard writer
    tb_writer.close()