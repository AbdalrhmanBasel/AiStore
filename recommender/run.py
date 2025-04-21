import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from src._0_data_preprocessing.preprocess import preprocess
from src._1_model_selection.GraphSAGEModelV0 import GraphSAGEModelV0
from src._2_training_model.training import train
from src._3_evaluating_model import evaluate_model
from src._4_recommenderation.recomender import generate_recommendations
from settings import *
import torch

# def train():
#     """
#     Function to handle the training of the model.
#     Includes steps like loading data, defining a model, training, and saving the model.
#     """
#     print("Training the model...")

#     # Load graph data with weights_only=False
#     train_data = torch.load(os.path.join(PROCESSED_DATA_DIR, "graph/positives/train_data.pt"), weights_only=False)
#     val_data = torch.load(os.path.join(PROCESSED_DATA_DIR, "graph/positives/val_data.pt"), weights_only=False)
#     train_neg_samples = torch.load(os.path.join(PROCESSED_DATA_DIR, "graph/negatives/train_neg_samples.pt"), weights_only=False)
#     val_neg_samples = torch.load(os.path.join(PROCESSED_DATA_DIR, "graph/negatives/val_neg_samples.pt"), weights_only=False)

#     # Define the model using settings
#     input_dim = 10 # Dynamically determine input dimensions
#     hidden_dim = HIDDEN_DIM
#     output_dim = OUTPUT_DIM
#     model = GraphSAGEModelV0(
#         input_dim=input_dim,
#         hidden_dim=hidden_dim,
#         output_dim=output_dim,
#         num_layers=NUM_LAYERS,
#         dropout=DROPOUT_RATE
#     )

#     # Define training parameters using settings
#     loss_fn = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy for link prediction
#     optimizer_name = "adam"
#     lr = LEARNING_RATE
#     epochs = EPOCHS
#     device = "cuda" if ENABLE_CUDA else "cpu"

#     # Early stopping configuration using settings
#     early_stopping_config = {'patience': PATIENCE, 'metric': 'loss'}
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pt")

#     # Start training
#     train_model(
#         model=model,
#         train_data=train_data,
#         val_data=val_data,
#         train_neg_samples=train_neg_samples,
#         val_neg_samples=val_neg_samples,
#         loss_fn=loss_fn,
#         optimizer_name=optimizer_name,
#         lr=lr,
#         epochs=epochs,
#         device=device,
#         early_stopping=early_stopping_config,
#         checkpoint_path=checkpoint_path
#     )

def evaluate():
    """
    Function to evaluate the trained model.
    Includes steps like model evaluation on test data and performance metrics calculation.
    """
    print("Evaluating the model...")

    # Load test data using paths from settings
    test_data = torch.load(os.path.join(PROCESSED_DATA_DIR, "graph/positives/test_data.pt"))
    test_neg_samples = torch.load(os.path.join(PROCESSED_DATA_DIR, "graph/negatives/test_neg_samples.pt"))

    # Load the trained model using settings
    model_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pt")
    model = torch.load(model_path)
    device = "cuda" if ENABLE_CUDA else "cpu"
    model.to(device)

    # Evaluate the model
    evaluate_model(
        model=model,
        test_data=test_data,
        test_neg_samples=test_neg_samples,
        device=device
    )


def recommend():
    """
    Function to generate recommendations.
    Includes the logic to load the trained model and make predictions on new data.
    """
    print("Generating recommendations...")

    # Load the trained model using settings
    model_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pt")
    model = torch.load(model_path)
    device = "cuda" if ENABLE_CUDA else "cpu"
    model.to(device)

    # Example: Generate recommendations for a specific user (e.g., user_id=12)
    graph_data = torch.load(GRAPH_SAVE_PATH)
    generate_recommendations(
        model=model,
        user_id=12,
        graph_data=graph_data,
        device=device
    )


def main():
    """
    Main entry point for the system.
    Orchestrates the execution of the full pipeline (preprocessing, training, evaluation, and recommendation).
    """
    print("Starting AI System...")

    # Step 1: Preprocess the data
    preprocess()

    # Step 2: Train the model
    train()

    # Step 3: Evaluate the model
    # evaluate()

    # # Step 4: Generate recommendations
    # recommend()


if __name__ == "__main__":
    # Run the main function to start the system
    main()