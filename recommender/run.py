import torch

from src._0_data_preprocessing.preprocess import preprocess
from src._2_training_model.utils.hyperparameter_tuning import run_hyperparameter_search
from src._2_training_model.training import training
from src._3_evaluating_model.evaluate_model import evaluate
from src._4_recommenderation.recomender import recommender

from settings import (
    HYPERPARAMETER_SEARCH,
    GRAPH_SAVE_PATH
)


def run():
    """
    Main entry point for the system.
    Orchestrates the execution of the full pipeline (preprocessing, training, evaluation, and recommendation).
    """
    print("Starting AI System...")

    # Step 1: Preprocess the data (uncomment if needed)
    # preprocess()

    # Step 2: Hyperparameter tuning (optional)
    # if HYPERPARAMETER_SEARCH:
    #     print("Running hyperparameter search...")
    #     run_hyperparameter_search()

    # Step 3: Train the model with best hyperparameters (uncomment if needed)
    # training()

    # Step 4: Evaluate the model
    # evaluate()

    # Step 5: Generate recommendations
    user_id = 0
    features = torch.rand((100, 16))  # Example features for 100 nodes
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Example edges
    train_edges = torch.tensor([[0, 1], [1, 2]])  # Example training edges

    top_k_indices, top_k_scores = recommender(user_id, features, edge_index, train_edges, top_k=5)
    print("Top-K Item Indices:", top_k_indices)
    print("Top-K Scores:", top_k_scores)






if __name__ == "__main__":
    run()
