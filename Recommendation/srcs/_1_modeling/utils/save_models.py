import torch
from logger import get_module_logger
import os
import sys

logger = get_module_logger("save_models")


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)

def save_model(model, user_embeddings, item_embeddings):
    # Save model weights
    torch.save(model.state_dict(), 'model.pth')
    
    # Save embeddings
    torch.save({
        'user_embeddings': user_embeddings,
        'item_embeddings': item_embeddings
    }, 'embeddings.pt')