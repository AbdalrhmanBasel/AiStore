import os
import sys
import torch
from logger import get_module_logger

logger = get_module_logger("load_best_model")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)


def load_best_model(model, load_dir="checkpoints/best", model_name="graphsage_best"):
    """
    Load the best model weights and embeddings from disk.

    Args:
        model (torch.nn.Module): Initialized model instance (weights will be loaded into it).
        load_dir (str): Directory where the best model is saved.
        model_name (str): Base filename used during saving.

    Returns:
        model (torch.nn.Module): Model with loaded weights.
        user_embeddings (torch.Tensor): Loaded user embeddings.
        item_embeddings (torch.Tensor): Loaded item embeddings.
    """
    model_path = os.path.join(load_dir, f"{model_name}_model.pth")
    embeddings_path = os.path.join(load_dir, f"{model_name}_embeddings.pt")

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        logger.info(f"Loaded model weights from {model_path}")

        embeddings = torch.load(embeddings_path, map_location=torch.device("cpu"))
        logger.info(f"Loaded embeddings from {embeddings_path}")

        return model, embeddings['user_embeddings'], embeddings['item_embeddings']
    except Exception as e:
        logger.error(f"Failed to load model or embeddings: {e}")
        raise
