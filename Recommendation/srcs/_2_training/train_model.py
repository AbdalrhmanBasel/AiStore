from logger import get_module_logger
import os
import sys
import torch 
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from srcs._3_evaluating.losses.bce_loss import generate_negative_samples, bce_loss
from srcs._3_evaluating.evaluate import evaluate

from settings import (
    EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    BATCH_SIZE,
    L2_REGULARIZATION,
    GRADIENT_CLIP,
    SCHEDULER,
    LR_GAMMA,
    LR_STEP_SIZE
)

logger = get_module_logger("train_model")


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)

def train_model(model, train_loader, val_loader, 
               epochs=EPOCHS, 
               patience=PATIENCE,
               lr=LEARNING_RATE,
               batch_size=BATCH_SIZE,
               weight_decay=L2_REGULARIZATION,
               gradient_clip=GRADIENT_CLIP):
    """
    Enhanced training loop with comprehensive logging and parameter integration
    
    Args:
        model: The GNN model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        lr: Learning rate
        batch_size: Batch size (for logging)
        weight_decay: L2 regularization strength
        gradient_clip: Gradient clipping threshold
    """
    # Log training configuration
    logger.info("🚀 Starting training with configuration:")
    logger.info(f"• Epochs: {epochs}")
    logger.info(f"• Batch size: {batch_size}")
    logger.info(f"• Learning rate: {lr}")
    logger.info(f"• L2 regularization: {weight_decay}")
    logger.info(f"• Gradient clipping: {gradient_clip}")
    logger.info(f"• Early stopping patience: {patience}")
    logger.info(f"• Scheduler: {SCHEDULER} (step_size={LR_STEP_SIZE}, gamma={LR_GAMMA})")
    
    best_ndcg = 0
    patience_counter = 0
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Configure learning rate scheduler
    if SCHEDULER == "StepLR":
        scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    elif SCHEDULER == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=LR_GAMMA)
    else:
        scheduler = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            pos_scores = model(batch.x, batch.edge_index)
            neg_edges = generate_negative_samples(batch.edge_index, batch.num_nodes)
            neg_scores = model(batch.x, neg_edges)
            
            # Calculate loss
            loss = bce_loss(pos_scores, neg_scores)
            loss.backward()
            
            # Gradient clipping
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
            # Log batch progress
            if total_batches % 10 == 0:
                logger.debug(f"Epoch {epoch+1} | Batch {total_batches} | Loss: {loss.item():.4f}")
        
        # Update learning rate
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.debug(f"Learning rate updated to: {current_lr:.2e}")
        
        # Validation
        avg_loss = total_loss / total_batches
        val_metrics = evaluate(model, val_loader)
        
        # Log epoch results
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val NDCG@10: {val_metrics['ndcg@k']:.4f} | "
            f"Precision@10: {val_metrics['precision@k']:.4f} | "
            f"Recall@10: {val_metrics['recall@k']:.4f}"
        )
        
        # Early stopping
        if val_metrics['ndcg@k'] > best_ndcg:
            best_ndcg = val_metrics['ndcg@k']
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
            logger.info(f"🔥 New best model! NDCG@10: {best_ndcg:.4f}")
        else:
            patience_counter += 1
            logger.debug(f"Early stopping counter: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            logger.info(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info("🎉 Training completed!")
    logger.info(f"Best validation NDCG@10: {best_ndcg:.4f}")
    return model