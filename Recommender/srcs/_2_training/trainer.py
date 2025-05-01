import os
import sys
import torch
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from logger import get_module_logger

from srcs._1_modeling.GraphSAGE import GraphSAGE
from srcs._2_training.train_model import train_model
from srcs._3_evaluating.evaluate import evaluate
from srcs._2_training.utils.save_models import save_model

from settings import (
    SAVED_MODEL_DIR,
    CHECKPOINT_DIR,
    MODEL_NAME,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    GRADIENT_CLIP,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    EDGE_INDEX_PATH,
    FEATURES_PATH,
    LABELS_PATH,
    BATCH_SIZE,
    HIDDEN_CHANNELS,
    OUT_CHANNELS,
    NUM_LAYERS,
    DROPOUT,
)

logger = get_module_logger("trainer")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def _load_graph_tensors():
    """Load the full graph tensors (edge_index, features, and labels) from disk."""
    edge_index = torch.load(EDGE_INDEX_PATH)
    x          = torch.load(FEATURES_PATH)
    labels     = torch.load(LABELS_PATH)
    return edge_index, x, labels


def _load_edge_splits():
    """Load pre‐split positive edge indices for train/val/test."""
    train_idx = torch.load(TRAIN_DATA_PATH)
    val_idx   = torch.load(VAL_DATA_PATH)
    test_idx  = torch.load(TEST_DATA_PATH)
    return train_idx, val_idx, test_idx


def _make_loader(edge_label_index, edge_index, x, batch_size):
    """
    Create a DataLoader for each edge split (train, validation, test).

    Args:
        edge_label_index (Tensor): The edge indices corresponding to the labels.
        edge_index (Tensor): The full graph's edge indices.
        x (Tensor): Node features.
        batch_size (int): Size of batches for training/validation/testing.

    Returns:
        DataLoader: A DataLoader for batching the data during training/validation/testing.
    """
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_label_index=edge_label_index
    )
    return DataLoader([data], batch_size=1, shuffle=True)


def trainer(device: str = "cpu"):
    """
    Orchestrates the full training pipeline for the GraphSAGE recommendation system.

    Steps:
        1) Load graph tensors and edge splits
        2) Build simple full-batch DataLoaders
        3) Initialize GraphSAGE model and optimizer
        4) Train with BPR loss and early stopping
        5) Evaluate on test split
        6) Save final model
    """
    logger.info("🚀 Initiating training pipeline...")

    # 1) Load graph tensors and splits
    edge_index, x, labels = _load_graph_tensors()
    train_idx, val_idx, test_idx = _load_edge_splits()
    logger.info(f"🔄 Loaded graph ({x.size(0)} nodes, {edge_index.size(1)} edges) and splits.")

    # 2) Wrap into DataLoaders
    train_loader = _make_loader(train_idx, edge_index, x, BATCH_SIZE)
    val_loader   = _make_loader(val_idx, edge_index, x, BATCH_SIZE)
    test_loader  = _make_loader(test_idx, edge_index, x, BATCH_SIZE)
    logger.info(f"✅ Data loaders created: "
                f"{len(train_loader.dataset)} train, "
                f"{len(val_loader.dataset)} val, "
                f"{len(test_loader.dataset)} test batches.")

    # 3) Initialize model & optimizer
    model = GraphSAGE(
        in_channels=x.size(1),
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logger.info("🛠️ Model and optimizer initialized.")

    # 4) Train the model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
        save_best=True,
        device=device
    )

    # 5) Evaluate on the test set
    logger.info("📊 Evaluating on test set...")
    metrics = evaluate(trained_model, test_loader, device=device)
    logger.info(f"✅ Test results — {metrics}")

    # 6) Save the final model
    save_model(
        model=trained_model,
        output_dir=SAVED_MODEL_DIR,
        model_name=MODEL_NAME
    )
    logger.info(f"💾 Final model saved to {SAVED_MODEL_DIR}/{MODEL_NAME}_model.pth")

    logger.info("🎯 Training pipeline completed.")

    return trained_model, metrics
