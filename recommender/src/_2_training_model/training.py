# training.py
import os
import sys
import torch
from src._2_training_model.train_model import train_model
from torch_geometric.loader import DataLoader  
from src._1_model_selection.GraphSAGEModelV0 import GraphSAGEModelV0
from src._0_data_preprocessing.utils.graph_dataset_loader import GraphDataset
from settings import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    DEVICE,
    PATIENCE,
    CHECKPOINT_DIR,
    MODEL_NAME,
    HIDDEN_DIM,
    OUTPUT_DIM,
    NUM_LAYERS,
    DROPOUT_RATE,
)


def training():
    """
    Function to handle the training of the model.
    Includes steps like loading data, defining a model, training, and saving the model.
    """
    print("Starting model training...")

    # Load datasets using GraphDataset
    train_dataset = GraphDataset(TRAIN_DATA_PATH)
    val_dataset = GraphDataset(VAL_DATA_PATH)

    # Use PyTorch Geometric's DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Dynamically determine input dimensions
    input_dim = train_dataset.num_node_features

    # Initialize the model
    model = GraphSAGEModelV0(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE,
    ).to(DEVICE)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Define checkpoint path
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pt")

    # Train the model
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=DEVICE,
        early_stopping={'patience': PATIENCE, 'metric': 'loss'},
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    training()