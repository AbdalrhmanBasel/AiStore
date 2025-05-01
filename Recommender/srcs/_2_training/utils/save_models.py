import os
import sys
import torch
from logger import get_module_logger

logger = get_module_logger("save_models")

from settings import EMBEDDINGS_SAVE_PATH, SAVED_MODEL_DIR, TRAINED_MODEL_PATH, MODEL_NAME

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)


def save_model_checkpoint(model, optimizer, epoch, path):
    """
    Save the model state_dict, optimizer state_dict, and epoch.
    """
    # Ensure the optimizer exists before saving it
    if optimizer:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
        }, path)

    logger.info(f"✅ Model saved at {path}.")


def save_final_model(model, optimizer, epoch, path):
    """
    Save the model state_dict, optimizer state_dict, and epoch.
    """
    # Ensure the optimizer exists before saving it
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if optimizer:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
        }, path)

    logger.info(f"✅ Model saved at {path}.")
    


def save_node_embeddings(model, graph, output_path=EMBEDDINGS_SAVE_PATH):
    """
    Extracts node embeddings using the trained model and saves them to a file.
    """
    logger.info("➡️  Saving node embeddings...")
    
    # Ensure that the model has a 'get_embeddings' method
    assert hasattr(model, 'get_embeddings'), "The model must have a 'get_embeddings' method."
    
    # Extract node embeddings from the model
    node_embeddings = model.get_embeddings(graph.x, graph.edge_index)
    
    # Save node embeddings to the specified path
    torch.save(node_embeddings, output_path)
    
    logger.info(f"✅ Node embeddings saved as '{output_path}'.")
