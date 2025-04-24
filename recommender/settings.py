# settings.py for Recommendation System (GNN-based)

import os
import torch
import numpy as np

# --------------------------------------------------------------
# General Settings
# --------------------------------------------------------------
PROJECT_NAME = "StoreX GNN Recommendation System"
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# --------------------------------------------------------------
# Dataset Settings
# --------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")
INTERIM_DATA_DIR = os.path.join(PROJECT_ROOT, "data/interim")

META_DATA_PATH = os.path.join(DATA_DIR, "raw/meta_Electronics.jsonl")
REVIEW_DATA_PATH = os.path.join(DATA_DIR, "raw/Electronics_2.csv")


GRAPH_SAVE_PATH = os.path.join(DATA_DIR, "processed/graph_data.pt")
EDGE_INDEX_PATH = os.path.join(GRAPH_SAVE_PATH, "processed/edge_index.pt")
FEATURES_PATH = os.path.join(GRAPH_SAVE_PATH, "processed/features.pt")
LABELS_PATH = os.path.join(GRAPH_SAVE_PATH, "processed/labels.pt")
FEATURES_MATRIX_PATH = os.path.join(INTERIM_DATA_DIR, "graph/feature_matrices/feature_matrix.csv")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOGGING_DIR = os.path.join(PROJECT_ROOT, "logs")
TENSORBOARD_LOG_DIR = os.path.join(LOGGING_DIR, "tensorboard")


SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data/interim/samples")
SAMPLE_META_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, "raw/metadata_small.jsonl")
SAMPLE_REVIEW_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, "raw/reviews_small.csv")

SAMPLE_CLEANED_META_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, "cleaned/cleaned_metadata_small.jsonl")
SAMPLE_CLEANED_REVIEW_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, "cleaned/cleaned_reviews_small.csv")

TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "graph/positives/train_data.pt")
VAL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "graph/positives/val_data.pt")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "graph/positives/test_data.pt")

SAMPLE_DATA_SIZE = 500
DATA_CHUNK_SIZE = 100

# --------------------------------------------------------------
# Model Settings
# --------------------------------------------------------------

MODEL_NAME = "GraphSAGEModelV0"  # Choose from available models like GraphSAGE, GCN, etc.
HIDDEN_DIM = 128          # Dimensionality of hidden layers
OUTPUT_DIM = 64           # Dimensionality of the output embeddings
NUM_LAYERS = 2            # Number of GNN layers (stacked)
ACTIVATION_FUNCTION = "relu"  # Activation function for the layers (options: "relu", "tanh", "sigmoid")

DROPOUT_RATE = 0.5        # Dropout rate for regularization

MODEL_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pt")
# --------------------------------------------------------------
# Training Settings
# --------------------------------------------------------------
BATCH_SIZE = 64           # Mini-batch size for training
LEARNING_RATE = 1e-3      # Learning rate for optimizer
EPOCHS = 1000               # Number of epochs to train

PATIENCE = 5              # Early stopping patience (if no improvement in validation performance)
L2_REGULARIZATION = 1e-5  # L2 regularization weight decay
MOMENTUM = 0.9            # Momentum for SGD (if applicable)
SCHEDULER = "StepLR"      # Scheduler type (options: "StepLR", "ExponentialLR", etc.)

LR_STEP_SIZE = 10         # Step size for learning rate decay (if using StepLR)
LR_GAMMA = 0.7            # Learning rate decay factor (if using StepLR)
GRADIENT_CLIP = 5.0       # Max norm for gradient clipping (set to None to disable)
 
# --------------------------------------------------------------
# Evaluation Settings
# --------------------------------------------------------------
EVALUATION_METRICS = ["precision", "recall", "ndcg"]  # Evaluation metrics
VALIDATION_SPLIT = 0.2   # Percentage of data to use for validation
EVALUATION_FREQUENCY = 5  # Evaluate the model every 5 epochs

# --------------------------------------------------------------
# Logging and Checkpointing
# --------------------------------------------------------------
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOGGING_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_SAVE_FREQUENCY = 10  # Save model checkpoints every 10 epochs
LOGGING_FREQUENCY = 1           # Log training progress every epoch

TENSORBOARD_LOG_DIR = os.path.join(LOGGING_DIR, "tensorboard")

# --------------------------------------------------------------
# Hardware Settings
# --------------------------------------------------------------
USE_GPU = True           # Set to False if no GPU is available
GPU_ID = 0               # ID of the GPU to use (if using multiple GPUs)
CPU_THREADS = 4          # Number of CPU threads to use for data loading

ENABLE_CUDA = USE_GPU and torch.cuda.is_available()  # Check if GPU is available

DEVICE = "cuda" if ENABLE_CUDA else "cpu"
# --------------------------------------------------------------
# Random Seed and Reproducibility
# --------------------------------------------------------------
SEED = 42                # Random seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------------------
# Hyperparameter Optimization (Optional)
# --------------------------------------------------------------
HYPERPARAMETER_SEARCH = True 
HYPERPARAMETER_EPOCHS = 5
SEARCH_ALGORITHM = "bayesian"  # Options: "grid", "random", "bayesian"
SEARCH_SPACE = {
    "HIDDEN_DIM": [64, 128, 256],
    "LEARNING_RATE": [1e-3, 1e-4, 1e-5],
    "DROPOUT_RATE": [0.2, 0.5, 0.7],
    "BATCH_SIZE": [32, 64, 128],
    "NUM_LAYERS": [2, 3, 4],
}
N_TRIALS = 50  
# --------------------------------------------------------------
# Miscellaneous
# --------------------------------------------------------------
USE_TQDM = True  # Use tqdm for progress bars during training and evaluation

# utils/print_settings.py
def print_settings(settings_module):
    """
    Print all non-callable attributes of the settings module.
    """
    print("\n🧠 AI System Settings Overview\n" + "-" * 40)
    for name in dir(settings_module):
        if name.startswith("__") or callable(getattr(settings_module, name)):
            continue  # Skip built-ins and functions
        value = getattr(settings_module, name)
        print(f"{name:30} : {value}")

# main.py or test_settings.py
if __name__ == "__main__":
    import settings
    print_settings(settings)