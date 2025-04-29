from logger import get_module_logger
import os
import sys
from srcs._3_evaluating.losses.bce_loss import generate_negative_samples, bce_loss
from srcs._3_evaluating.evaluate import evaluate
from srcs._2_training.train_model import train_model
from srcs._1_modeling.GraphSAGE import GraphSAGE
from srcs._2_training.data_loaders import create_data_loaders

logger = get_module_logger("trainer")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)

def trainer(graph, mappings, train_edge_index, val_edge_index, test_edge_index):
    logger.info("🔄 Starting training process.")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(graph)
    
    # Instantiate model
    model = GraphSAGE(
        in_channels=graph.num_node_features,
        hidden_channels=64,
        out_channels=32
    )

    # Train model
    trained_model = train_model(model, train_loader, val_loader)

    logger.info("✅ Training process completed.")
