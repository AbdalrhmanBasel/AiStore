import os
import torch
from src._2_training_model.train_model import train_model
from torch_geometric.loader import DataLoader  
from src._1_model_selection.GraphSAGEModelV0 import GraphSAGEModelV0
from src._0_data_preprocessing.graph_construction.GraphDataset import GraphDataset
from src._2_training_model.utils.load_hyperparameters import load_best_hyperparameters
from settings import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    DEVICE,
    PATIENCE,
    CHECKPOINT_DIR,
    MODEL_NAME,
    HYPERPARAMETERS_PATH
)


def training():
    """
    Function to handle the training of the model.
    Includes steps like loading data, defining a model, training, and saving the model.
    """
    print("Starting model training...")

    # Load best hyperparameters from file
    best_hyperparameters_file = HYPERPARAMETERS_PATH
    hyperparams = load_best_hyperparameters(best_hyperparameters_file)
    print(f"Loaded Hyperparameters: {hyperparams}")

    # Extract hyperparameters
    HIDDEN_DIM = hyperparams["HIDDEN_DIM"]
    LEARNING_RATE = hyperparams["LEARNING_RATE"]
    DROPOUT_RATE = hyperparams["DROPOUT_RATE"]
    BATCH_SIZE = hyperparams["BATCH_SIZE"]
    NUM_LAYERS = hyperparams["NUM_LAYERS"]

    # Load datasets using GraphDataset
    train_dataset = GraphDataset(TRAIN_DATA_PATH)
    val_dataset = GraphDataset(VAL_DATA_PATH)

    # Use PyTorch Geometric's DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Dynamically determine input dimensions
    input_dim = train_dataset.num_node_features

    # Initialize the model with loaded hyperparameters
    model = GraphSAGEModelV0(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=1,  # Assuming binary classification (output_dim=1)
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
        epochs=100,  # You can also load EPOCHS from hyperparameters if needed
        device=DEVICE,
        early_stopping={'patience': PATIENCE, 'metric': 'loss'},
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    training()