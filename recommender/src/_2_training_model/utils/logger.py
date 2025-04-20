# logger.py
import logging
from torch.utils.tensorboard import SummaryWriter

# Set up a single logger instance
logger = logging.getLogger("train_logger")  # Use a global, unique name
logger.setLevel(logging.INFO)

# Avoid adding multiple handlers in case of repeated imports
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# TensorBoard writer (used globally)
tb_writer = SummaryWriter()
