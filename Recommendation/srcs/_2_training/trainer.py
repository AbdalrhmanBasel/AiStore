from logger import get_module_logger
from srcs._3_evaluating.evaluate import evaluate
from srcs._2_training.train_model import train_model
from srcs._2_training.data_loaders import create_data_loaders
from srcs._1_modeling.GraphSAGE import GraphSAGE
import torch.optim as optim
import os
import sys

logger = get_module_logger("trainer")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from settings import LEARNING_RATE, EPOCHS

def trainer(graph, mappings, *args, **kwargs):
    """
    Orchestrates training and evaluation of the GraphSAGE recommendation model.
    Ignores explicit edge_index splits since loaders use graph attributes.
    """
    logger.info("🔄 Starting training process.")

    train_loader, val_loader, test_loader = create_data_loaders(graph)

    model = GraphSAGE(
        in_channels=graph.num_node_features,
        hidden_channels=64,
        out_channels=32
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=EPOCHS,
        checkpoint_dir="checkpoints",
        save_best=True,
        device="cpu"
    )

    test_results = evaluate(trained_model, test_loader)
    logger.info("✅ Training process completed.")
    logger.info("🔍 Evaluating the trained model on the test set.")
    logger.info(f"Test Results: {test_results}")

    return trained_model
