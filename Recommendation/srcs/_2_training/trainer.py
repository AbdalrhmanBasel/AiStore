from logger import get_module_logger
import os
import sys
import torch 
from srcs._3_evaluating.losses.bce_loss import generate_negative_samples, bce_loss
from srcs._3_evaluating.evaluate import evaluate
from srcs._2_training.train_model import train_model


logger = get_module_logger("trainer")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)


def trainer():
    logger.info("🔄 Starting training process.")
    # create_data_loaders
    # train_model()
    logger.info("✅ Training process completed.")
















