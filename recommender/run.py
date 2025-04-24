import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from src._0_data_preprocessing.preprocess import preprocess
from src._2_training_model.utils.hyperparameter_tuning import run_hyperparameter_search
# from src._1_model_selection.GraphSAGEModelV0 import GraphSAGEModelV0
from src._2_training_model.training import training
from src._3_evaluating_model.evaluate_model import evaluate
# from src._4_recommenderation.recomender import generate_recommendations
# from src._2_training_model.utils.hyperparameter_tuning import run_hyperparameter_search
import torch

from settings import (
        HYPERPARAMETER_SEARCH,
        PROCESSED_DATA_DIR,
        CHECKPOINT_DIR,
        MODEL_NAME,
        ENABLE_CUDA,
        GRAPH_SAVE_PATH
    )




# def recommend():
#     """
#     Function to generate recommendations.
#     Includes the logic to load the trained model and make predictions on new data.
#     """
#     print("Generating recommendations...")

#     # Load the trained model using settings
#     model_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pt")
#     model = torch.load(model_path)
#     device = "cuda" if ENABLE_CUDA else "cpu"
#     model.to(device)

#     # Example: Generate recommendations for a specific user (e.g., user_id=12)
#     graph_data = torch.load(GRAPH_SAVE_PATH)
#     generate_recommendations(
#         model=model,
#         user_id=12,
#         graph_data=graph_data,
#         device=device
#     )


import torch
from torch_geometric.loader import DataLoader
from src._1_model_selection.GraphSAGEModelV0 import GraphSAGEModelV0
from src._3_evaluating_model.evaluate_model import evaluate
from src._0_data_preprocessing.graph_construction.GraphDataset import GraphDataset
from src._4_recommenderation.recomender import generate_recommendations




def run():
    """
    Main entry point for the system.
    Orchestrates the execution of the full pipeline (preprocessing, training, evaluation, and recommendation).
    """
    print("Starting AI System...")
    
    # Step 1: Preprocess the data
    # preprocess()

    # Step 2: Hyperparameter tuning (optional)
    # if HYPERPARAMETER_SEARCH:
    #     print("Running hyperparameter search...")
    #     run_hyperparameter_search()

    # Step 3: Train the model with best hyperparameters
    # training()

    # Step 4: Evaluate the model
    evaluate()
    
    # # Step 5: Generate recommendations
    # user_id = 42  # Replace with a valid user ID
    # graph_data = torch.load(GRAPH_SAVE_PATH, weights_only=False)
    # if user_id >= graph_data.num_users:
    #     raise ValueError(f"User ID {user_id} is out of range. Maximum user ID is {graph_data.num_users - 1}.")
    # generate_recommendations(user_id=user_id)

if __name__ == "__main__":
    run()

