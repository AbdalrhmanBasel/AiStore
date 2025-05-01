import os
import sys
import torch
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from logger import get_module_logger
from srcs._2_training.utils.save_models import save_model
from settings import (
    MODEL_NAME,
    EPOCHS,
    PATIENCE,
    GRADIENT_CLIP,
)

logger = get_module_logger("train_model")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)



@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    total_metric = 0.0
    for batch in loader:
        batch = batch.to(device)
        pos_idx = batch.edge_label_index
        neg_idx = negative_sampling(pos_idx, batch.num_nodes, pos_idx.size(1))
        pos_scores, neg_scores = model(batch.x, batch.edge_index, pos_idx, neg_idx)
        total_metric += -F.logsigmoid(pos_scores - neg_scores).mean().item()
    return total_metric / len(loader)

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs: int = EPOCHS,
    checkpoint_dir: str = "checkpoints",
    save_best: bool = True,
    device: str = "cpu"
):
    model.to(device)
    best_val_loss = float('inf')  
    epochs_no_improve = 0 
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0

        # Training loop
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pos_idx = batch.edge_label_index
            neg_idx = negative_sampling(pos_idx, batch.num_nodes, pos_idx.size(1))

            pos_scores, neg_scores = model(batch.x, batch.edge_index, pos_idx, neg_idx)

            # BPR loss calculation
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            loss.backward()

            # Gradient clipping
            if GRADIENT_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # --- Validation & Early Stopping ---
        val_metric = evaluate(model, val_loader, device)

        # Log epoch information
        epoch_log = f"Epoch {epoch}/{num_epochs}: Train BPR-Loss = {avg_loss:.4f}, Val BPR-Loss = {val_metric:.4f}"

        # --- Save model if validation loss improves ---
        if val_metric < best_val_loss:
            best_val_loss = val_metric
            epochs_no_improve = 0  # Reset early stopping counter

        # Save the model only when there's an improvement in validation loss
        if save_best:
            save_model(model, output_dir=checkpoint_dir, model_name=MODEL_NAME)  # Removed user_embeddings and item_embeddings
            epoch_log += f", Best Val Model (Val BPR-Loss = {best_val_loss:.4f}), Model Saved"


        logger.info(epoch_log)

        # Early stopping if no improvement for PATIENCE epochs
        if epochs_no_improve >= PATIENCE:
            logger.info("⚠️ Early stopping triggered.")
            break

    return model
