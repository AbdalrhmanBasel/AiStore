import os
import sys
import torch
from logger import get_module_logger

logger = get_module_logger("load_models")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from settings import MODEL_NAME


def load_model(model, load_dir, model_name=MODEL_NAME, load_embeddings=False, device='cpu'):
    """
    Load model weights and optionally user/item embeddings.

    Args:
        model (torch.nn.Module): Model instance to load weights into.
        load_dir (str): Directory containing the saved model and embeddings.
        model_name (str): Base name for the saved files.
        load_embeddings (bool): Whether to load and return embeddings.
        device (str): 'cpu' or 'cuda'

    Returns:
        If load_embeddings:
            Tuple(model, user_embeddings, item_embeddings)
        Else:
            model
    """
    model_path = os.path.join(load_dir, f"{model_name}_model.pth")
    embeddings_path = os.path.join(load_dir, f"{model_name}_embeddings.pt")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    if load_embeddings:
        if not os.path.exists(embeddings_path):
            logger.error(f"Embeddings file not found at {embeddings_path}")
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        try:
            data = torch.load(embeddings_path, map_location=device)
            logger.info(f"Embeddings loaded from {embeddings_path}")
            return model, data['user_embeddings'], data['item_embeddings']
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    return model
