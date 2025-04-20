# settings.py for Recommendation System (GNN-based)

import os
import torch
import numpy as np

# --------------------------------------------------------------
# General settings
# --------------------------------------------------------------
PROJECT_NAME = "StoreX GNN Recommendation System"
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# --------------------------------------------------------------
# Dataset Settings
# --------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
META_DATA_PATH = os.path.join(DATA_DIR, "metadata_electronics_small.jsonl")  # Path to product metadata
REVIEW_DATA_PATH = os.path.join(DATA_DIR, "reviews_electronics_small.csv")  # Path to user reviews data
# LINKS_DATA_PATH = os.path.join(DATA_DIR, "links.csv")  # Path to user-item interaction data

SAMPLE_DATA_SIZE = 900_000
DATA_CHUNK_SIZE = 10_000

# SAMPLE_DATA_SIZE = 100
# DATA_CHUNK_SIZE = 10

# Graph Construction Settings
GRAPH_SAVE_PATH = os.path.join(DATA_DIR, "graph_data")
EDGE_INDEX_PATH = os.path.join(GRAPH_SAVE_PATH, "edge_index.pt")  # Path for saving edge indices
FEATURES_PATH = os.path.join(GRAPH_SAVE_PATH, "features.pt")  # Path for saving node features
LABELS_PATH = os.path.join(GRAPH_SAVE_PATH, "labels.pt")  # Path for saving labels

# --------------------------------------------------------------
# Model Settings
# --------------------------------------------------------------
MODEL_NAME = "GraphSAGE"  # Choose from available models like GraphSAGE, GCN, etc.
HIDDEN_DIM = 128  # Dimensionality of hidden layers
OUTPUT_DIM = 64  # Dimensionality of the output embeddings
NUM_LAYERS = 2  # Number of GNN layers (stacked)
ACTIVATION_FUNCTION = "relu"  # Activation function for the layers (options: "relu", "tanh", "sigmoid")

# Dropout settings for regularization
DROPOUT_RATE = 0.5

# --------------------------------------------------------------
# Training Settings
# --------------------------------------------------------------
BATCH_SIZE = 64  # Mini-batch size for training
LEARNING_RATE = 1e-3  # Learning rate for optimizer
EPOCHS = 50  # Number of epochs to train
PATIENCE = 5  # Early stopping patience (if no improvement in validation performance)
L2_REGULARIZATION = 1e-5  # L2 regularization weight decay

# Optimizer settings
# OPTIMIZER = "Adam"  # Options: "Adam", "SGD", etc.
MOMENTUM = 0.9  # Momentum for SGD (if applicable)

# Loss function settings
# LOSS_FUNCTION = "BCEWithLogitsLoss"  # Binary Cross Entropy Loss (for binary classification tasks)

# Learning rate scheduler
SCHEDULER = "StepLR"  # Scheduler type (options: "StepLR", "ExponentialLR", etc.)
LR_STEP_SIZE = 10  # Step size for learning rate decay (if using StepLR)
LR_GAMMA = 0.7  # Learning rate decay factor (if using StepLR)

# --------------------------------------------------------------
# Evaluation Settings
# --------------------------------------------------------------
EVALUATION_METRICS = ["precision", "recall", "ndcg"]  # Evaluation metrics (can include precision, recall, etc.)
VALIDATION_SPLIT = 0.2  # Percentage of data to use for validation

# Evaluation frequency (how often to evaluate the model)
EVALUATION_FREQUENCY = 5  # Evaluate the model every 5 epochs

# --------------------------------------------------------------
# Logging and Checkpointing
# --------------------------------------------------------------
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints")
LOGGING_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
CHECKPOINT_SAVE_FREQUENCY = 10  # Save model checkpoints every 10 epochs
LOGGING_FREQUENCY = 1  # Log training progress every epoch

# Tensorboard for visualizing training progress
TENSORBOARD_LOG_DIR = os.path.join(LOGGING_DIR, "tensorboard")

# --------------------------------------------------------------
# Hardware Settings
# --------------------------------------------------------------
USE_GPU = True  # Set to False if no GPU is available
GPU_ID = 0  # ID of the GPU to use (if using multiple GPUs)
CPU_THREADS = 4  # Number of CPU threads to use for data loading

# --------------------------------------------------------------
# Random Seed and Reproducibility
# --------------------------------------------------------------
SEED = 42  # Random seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------------------
# Hyperparameter Optimization (Optional)
# --------------------------------------------------------------
HYPERPARAMETER_SEARCH = False  # Set to True to use hyperparameter search (e.g., grid search, random search)
SEARCH_ALGORITHM = "grid"  # Options: "grid", "random", "bayesian"
SEARCH_SPACE = {
    "HIDDEN_DIM": [64, 128, 256],
    "LEARNING_RATE": [1e-3, 1e-4, 1e-5],
    "DROPOUT_RATE": [0.2, 0.5, 0.7],
}

# --------------------------------------------------------------
# Miscellaneous
# --------------------------------------------------------------
USE_TQDM = True  # Use tqdm for progress bars during training and evaluation
ENABLE_CUDA = USE_GPU and torch.cuda.is_available()  # Check if GPU is available
