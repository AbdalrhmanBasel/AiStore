import logging
import os
from colorama import Fore, Style, init
from settings import SESSION_LOG_DIR

os.makedirs(SESSION_LOG_DIR, exist_ok=True)

init(autoreset=True)

LOG_COLORS = {
    logging.DEBUG: Fore.BLUE,
    logging.INFO: Fore.CYAN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_COLORS.get(record.levelno, "")
        reset = Style.RESET_ALL
        formatted = super().format(record)
        return f"{color}{formatted}{reset}"

def get_module_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        return logger

    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    log_file_path = os.path.join(SESSION_LOG_DIR, f"{module_name}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter(file_formatter._fmt, file_formatter.datefmt))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
