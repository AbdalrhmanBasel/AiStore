import torch
from logger import logger

def get_best_device(prefer_gpu: bool = True, verbose: bool = True) -> torch.device:
    """
    Automatically selects the best available device for PyTorch training.

    Args:
        prefer_gpu (bool): If True, prioritizes CUDA (NVIDIA GPU) and MPS (Apple GPU).
        verbose (bool): If True, logs the selected device.

    Returns:
        torch.device: The best available device ('cuda', 'mps', or 'cpu').
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif prefer_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        logger.info(f"Using device: {device}")

    return device
