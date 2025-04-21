import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any
from src._2_training_model.utils.checkpoints import save_checkpoint, load_checkpoint
from src._1_model_selection.GraphSAGEModelV0 import GraphSAGEModelV0
from src._3_evaluating_model.evaluate_model import evaluate_model
from src._3_evaluating_model.LinkPredictionLoss import LinkPredictionLoss
from src._2_training_model.utils.logger import setup_logger
from src._0_data_preprocessing.utils.graph_dataset_loader import GraphDataset
from settings import * 
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)

# Setup logger
logger = setup_logger(__name__)

# TensorBoard Setup
tb_writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)


def train_step(
    model: nn.Module,
    data: Any,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_clip: Optional[float] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> torch.Tensor:
    """
    Performs a single training step with mixed precision support (autocast) for efficiency.
    """
    model.train()

    # Get node features (x), edge index (edge_index), and labels (y)
    x, edge_index, y = data.x, data.edge_index, data.y

    # Zero gradients
    optimizer.zero_grad()

    # Mixed precision (use autocast for efficient training)
    with torch.cuda.amp.autocast(enabled=True):  # Enable mixed-precision for forward pass
        output = model(x, edge_index)
        loss = loss_fn(output, y)

    # Backward pass
    if scaler:  # Mixed precision support
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # Gradient clipping (if specified)
    if gradient_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

    # Update parameters
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return loss


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    device: str = "cpu",
    scheduler: Optional[_LRScheduler] = None,
    weight_decay: float = 0.0,
    momentum: float = MOMENTUM,
    gradient_clip: Optional[float] = None,
    early_stopping: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[str] = None,
    resume_from_checkpoint: bool = False,
    print_interval: int = 10,
) -> None:
    """
    A customizable training loop with mixed precision support, early stopping, and TensorBoard logging.
    Supports model checkpoint saving/loading, optimizer and scheduler state saving/loading.
    """
    # Set device and move model to the correct device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize mixed-precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Initialize or load the scheduler
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

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

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        # Iterate over training data
        for data in train_dataloader:
            data = data.to(device)  # Move data to device

            # Perform training step
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
        if (epoch + 1) % print_interval == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
            if val_dataloader:
                logger.info(f"Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if early_stopping:
            metric = epoch_loss if early_stopping["metric"] == "loss" else 0  # Add support for custom metrics
            if metric < best_metric:
                best_metric = metric
                epochs_without_improvement = 0

                # Save checkpoint
                if checkpoint_path:
                    save_checkpoint(model, optimizer, epoch + 1, epoch_loss, checkpoint_path, scheduler)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} due to no improvement in {early_stopping['metric']}")
                    break

    logger.info("Training completed.")

    # Final model save
    if checkpoint_path:
        save_checkpoint(model, optimizer, epochs, epoch_loss, checkpoint_path, scheduler)

    # Close TensorBoard writer
    tb_writer.close()


from torch_geometric.data import DataLoader  # Use PyTorch Geometric's DataLoader

def train():
    """
    Function to handle the training of the model.
    Includes steps like loading data, defining a model, training, and saving the model.
    """
    logger.info("Starting model training...")
    
    # Prepare dataset and dataloaders
    train_dataset = GraphDataset(TRAIN_DATA_PATH)
    val_dataset = GraphDataset(VAL_DATA_PATH)

    # Use PyTorch Geometric's DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    feature_matrix = pd.read_csv(FEATURES_MATRIX_PATH)
    INPUT_DIM = feature_matrix.shape[1]

    # Initialize the model
    model = GraphSAGEModelV0(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE,
    )

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)

    # Define the loss function
    loss_fn = LinkPredictionLoss()

    # Training the model
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS,
        device="cuda" if ENABLE_CUDA else "cpu",
        scheduler=None,  # You can pass a learning rate scheduler if you want
        gradient_clip=GRADIENT_CLIP,
        early_stopping={'patience': PATIENCE, 'metric': 'loss'},
        checkpoint_path=os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pt"),
        resume_from_checkpoint=False,
    )


if __name__ == "__main__":
    train()