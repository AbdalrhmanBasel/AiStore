import os
import sys
import torch
from logger import get_module_logger

logger = get_module_logger("save_models")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from settings import MODEL_NAME, SAVED_MODEL_PATH


def save_checkpoint(model, user_embeddings, item_embeddings, checkpoint_dir="checkpoints", model_name=MODEL_NAME):
    """
    Save model and embeddings to checkpoints directory (intermediate training state).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = os.path.join(checkpoint_dir, f"{model_name}_model.pth")
    embeddings_path = os.path.join(checkpoint_dir, f"{model_name}_embeddings.pt")

    try:
        torch.save(model.state_dict(), model_path)
        torch.save({
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings
        }, embeddings_path)

        logger.info(f"Checkpoint model saved to {model_path}")
        logger.info(f"Checkpoint embeddings saved to {embeddings_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise


def save_model(model, user_embeddings, item_embeddings, output_dir=SAVED_MODEL_PATH, model_name=MODEL_NAME):
    """
    Save final model and embeddings to production artifacts directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{model_name}_model.pth")
    embeddings_path = os.path.join(output_dir, f"{model_name}_embeddings.pt")

    try:
        torch.save(model.state_dict(), model_path)
        torch.save({
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings
        }, embeddings_path)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Embeddings saved to {embeddings_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")
        raise
