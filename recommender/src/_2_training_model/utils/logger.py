# utils/logger_utils.py
import logging

def setup_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger with a consistent format.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger