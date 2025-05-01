import torch

def load_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, checkpoint_path: str = "checkpoints/best_model.pt", device: str = "cpu"):
    """
    Load the model (and optionally optimizer and epoch) from a checkpoint.

    Args:
        model (torch.nn.Module): The model to load state_dict into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state_dict into.
        checkpoint_path (str): Path to the checkpoint to load.
        device (str): Device to load the model on (e.g., "cpu", "cuda").

    Returns:
        model (torch.nn.Module): The model with loaded state_dict.
        optimizer (torch.optim.Optimizer, optional): The optimizer with loaded state_dict, if provided.
        epoch (int, optional): The epoch number, if saved.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state (optional)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load epoch (optional)
    epoch = checkpoint.get("epoch", None)

    print(f"🔄 Model loaded from {checkpoint_path}.")
    if epoch is not None:
        print(f"🔄 Resumed from epoch {epoch}.")
    
    return model, optimizer, epoch
