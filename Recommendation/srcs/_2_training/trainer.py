import os
import sys
import torch.optim as optim
from logger import get_module_logger

from srcs._1_modeling.GraphSAGE import GraphSAGE
from srcs._2_training.data_loaders import create_data_loaders
from srcs._2_training.train_model import train_model
from srcs._3_evaluating.evaluate import evaluate
from srcs._1_modeling.utils.save_models import save_model
from settings import LEARNING_RATE, EPOCHS, MODEL_NAME, SAVED_MODEL_PATH

logger = get_module_logger("trainer")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def trainer(graph, mappings, *args, **kwargs):
    """
    Orchestrates full training pipeline for the GraphSAGE recommendation system.
    Includes training, validation, testing, and saving final model.
    """
    logger.info("🚀 Initiating training pipeline...")

    # Step 1: Create Data Loaders
    train_loader, val_loader, test_loader = create_data_loaders(graph)

    # Step 2: Initialize Model & Optimizer
    model = GraphSAGE(
        in_channels=graph.num_node_features,
        hidden_channels=64,
        out_channels=32
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Step 3: Train Model
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

    # Step 4: Evaluate on Test Set
    logger.info("📊 Evaluating model on test set...")
    test_results = evaluate(trained_model, test_loader)
    logger.info(f"✅ Test Evaluation Complete — Results: {test_results}")

    # Step 5: Save Final Model
    logger.info("💾 Saving final trained model to disk...")
    save_model(
        model=trained_model,
        user_embeddings=None,
        item_embeddings=None,
        output_dir=SAVED_MODEL_PATH,
        model_name=MODEL_NAME
    )
    logger.info(f"📁 Final model saved to {SAVED_MODEL_PATH} as {MODEL_NAME}_model.pth")

    logger.info("🎯 Training pipeline completed.")
    return trained_model
