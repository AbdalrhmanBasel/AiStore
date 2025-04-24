# train_model.py
import os
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader  
from typing import Optional, Dict, Any
from src._2_training_model.utils.checkpoints import save_checkpoint, load_checkpoint
from src._2_training_model.train_step import train_step
from src._3_evaluating_model.evaluate_model import evaluate_training_model
from settings import (
    CHECKPOINT_DIR,
    TENSORBOARD_LOG_DIR,
    LEARNING_RATE,
    EPOCHS,
    GRADIENT_CLIP,
    PATIENCE,
    MODEL_NAME,
    DEVICE,
    PROJECT_ROOT
)

checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pt")

def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader],
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    device: str = "cpu",
    scheduler: Optional[_LRScheduler] = None,
    gradient_clip: Optional[float] = GRADIENT_CLIP,
    early_stopping: Optional[Dict[str, Any]] = {"patience": PATIENCE, "metric": "loss"},
    checkpoint_path: Optional[str] = checkpoint_path,
    resume_from_checkpoint: bool = False,
) -> None:
    """
    A customizable training loop with mixed precision support, early stopping, and TensorBoard logging.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = torch.cuda.amp.GradScaler()

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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

    tb_writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        for data in train_dataloader:
            data = data.to(device)  # Move data to device
            loss = train_step(model, data, loss_fn, optimizer, gradient_clip, scaler)
            epoch_loss += loss.item()

        # Average loss over the epoch
        epoch_loss /= len(train_dataloader)

        tb_writer.add_scalar('Loss/train', epoch_loss, epoch)

        if val_dataloader:
            val_loss = evaluate_training_model(model, val_dataloader, loss_fn, device)
            tb_writer.add_scalar('Loss/val', val_loss, epoch)

        scheduler.step()

        # Logging progress
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        if val_dataloader:
            print(f"Validation Loss: {val_loss:.4f}")

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
                    print(f"Early stopping at epoch {epoch + 1} due to no improvement in {early_stopping['metric']}")
                    break

    print("Training completed.")

    if checkpoint_path:
        save_checkpoint(model, optimizer, epochs, epoch_loss, checkpoint_path, scheduler)

    tb_writer.close()



def train_model(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = EPOCHS,
    device: str = DEVICE,
    early_stopping: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[str] = None,
):
    """
    A customizable training loop with mixed precision support, early stopping, and TensorBoard logging.
    """
    # Set device and move model to the correct device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize mixed-precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Initialize TensorBoard writer
    tb_writer = torch.utils.tensorboard.SummaryWriter()

    # Ensure the checkpoint directory exists
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Initialize early stopping
    if early_stopping:
        best_metric = float("inf") if early_stopping["metric"] == "loss" else -float("inf")
        patience = early_stopping["patience"]
        epochs_without_improvement = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Train on batches
        for data in train_dataloader:
            data = data.to(device)
            loss = train_step(model, data, loss_fn, optimizer, gradient_clip=None, scaler=scaler)
            epoch_loss += loss.item()

        # Average loss over the epoch
        epoch_loss /= len(train_dataloader)
        tb_writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Validation
        if val_dataloader:
            val_metrics = evaluate_training_model(model, val_dataloader, loss_fn, device)
            val_loss = val_metrics["loss"]  # Extract validation loss from the dictionary
            tb_writer.add_scalar('Loss/val', val_loss, epoch)  # Log validation loss

            # Log other metrics (optional)
            tb_writer.add_scalar('Metrics/precision', val_metrics["precision"], epoch)
            tb_writer.add_scalar('Metrics/recall', val_metrics["recall"], epoch)
            tb_writer.add_scalar('Metrics/ndcg', val_metrics["ndcg"], epoch)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping logic
        if early_stopping and val_dataloader:
            metric = val_loss if early_stopping["metric"] == "loss" else val_metrics["ndcg"]
            if (early_stopping["metric"] == "loss" and metric < best_metric) or \
               (early_stopping["metric"] != "loss" and metric > best_metric):
                best_metric = metric
                epochs_without_improvement = 0
                if checkpoint_path:
                    save_checkpoint(model, optimizer, epoch + 1, epoch_loss, checkpoint_path, None)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1} due to no improvement in {early_stopping['metric']}")
                    break
                

    # Final model save
    if checkpoint_path:
        save_checkpoint(model, optimizer, epochs, epoch_loss, checkpoint_path, None)

    # Save the final model to artifacts
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts", "models")
    os.makedirs(artifacts_dir, exist_ok=True)
    final_model_path = os.path.join(artifacts_dir, f"{MODEL_NAME}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Close TensorBoard writer
    tb_writer.close()