# hyperparameter_tuning.py
import os
import optuna
import torch
from src._2_training_model.training import train_model
from src._3_evaluating_model.evaluate_model import evaluate_training_model
from src._1_model_selection.GraphSAGEModelV0 import GraphSAGEModelV0
from src._0_data_preprocessing.graph_construction.GraphDataset import GraphDataset
from torch_geometric.loader import DataLoader
from settings import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    HYPERPARAMETER_EPOCHS,
    DEVICE,
    ARTIFFACTS_PATH,
    HYPERPARAMETER_SEARCH,
    N_TRIALS,
    OUTPUT_DIM,
    PATIENCE
)

def objective(trial):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_int("HIDDEN_DIM", 64, 256)
    learning_rate = trial.suggest_float("LEARNING_RATE", 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float("DROPOUT_RATE", 0.1, 0.7)
    batch_size = trial.suggest_categorical("BATCH_SIZE", [16, 32, 64, 128])
    num_layers = trial.suggest_int("NUM_LAYERS", 2, 4)

    # Load datasets
    train_dataset = GraphDataset(TRAIN_DATA_PATH)
    val_dataset = GraphDataset(VAL_DATA_PATH)

    # Use PyTorch Geometric's DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the model
    input_dim = train_dataset.num_node_features
    model = GraphSAGEModelV0(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=OUTPUT_DIM,
        num_layers=num_layers,
        dropout=dropout_rate,
    ).to(DEVICE)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train the model
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=HYPERPARAMETER_EPOCHS,
        device=DEVICE,
        early_stopping={'patience': PATIENCE, 'metric': 'loss'},
        checkpoint_path=None,  # No checkpoint saving during hyperparameter search
    )

    # Evaluate the model on validation data
    val_metrics = evaluate_training_model(model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=DEVICE)
    return val_metrics["loss"]


def run_hyperparameter_search():
    """
    Run hyperparameter search using Optuna.
    """
    if not HYPERPARAMETER_SEARCH:
        print("Hyperparameter search is disabled in settings.")
        return

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")  # Minimize validation loss
    study.optimize(objective, n_trials=N_TRIALS)

    # Print the best trial results
    print("\nBest Trial:")
    best_trial = study.best_trial
    print(f"  Validation Loss: {best_trial.value}")
    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save the best parameters to a file
    best_params_path = os.path.join(ARTIFFACTS_PATH, "hyperparmeters/best_hyperparameters.txt")
    with open(best_params_path, "w") as f:
        f.write("Best Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"{key}: {value}\n")
    print(f"\nBest hyperparameters saved to: {best_params_path}")


if __name__ == "__main__":
    run_hyperparameter_search()