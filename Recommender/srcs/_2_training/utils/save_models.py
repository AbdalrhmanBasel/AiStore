import os
import torch
from logger import get_module_logger

logger = get_module_logger("save_models")


def save_model(model, output_dir, model_name):
    """Save the model state dict to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    # print(f"Model saved at {model_path}")
