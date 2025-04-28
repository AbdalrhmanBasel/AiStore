# settings.py for Recommendation System (GNN-based)

import os
from datetime import datetime


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


COMPLETE_GRAPH_SAVE_PATH = os.path.join(DATA_DIR, "processed/complete_graph")
EDGE_INDEX_PATH = os.path.join(COMPLETE_GRAPH_SAVE_PATH, "edge_index.pt")
FEATURES_PATH = os.path.join(COMPLETE_GRAPH_SAVE_PATH, "features.pt")
LABELS_PATH = os.path.join(COMPLETE_GRAPH_SAVE_PATH, "labels.pt")
MAPPING_GRAPH_PATH = os.path.join(COMPLETE_GRAPH_SAVE_PATH, "mappings.pkl")
FEATURES_MATRIX_PATH = os.path.join(INTERIM_DATA_DIR, "graph/feature_matrices/feature_matrix.csv")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOGGING_DIR = os.path.join(PROJECT_ROOT, "logs")
TENSORBOARD_LOG_DIR = os.path.join(LOGGING_DIR, "tensorboard")


SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "data/interim/samples")
SAMPLE_META_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, "raw/metadata_small.jsonl")
SAMPLE_REVIEW_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, "raw/reviews_small.csv")

SAMPLE_CLEANED_META_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, "raw/cleaned_meta_dataset.jsonl")
SAMPLE_NLP_CLEANED_META_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, "cleaned/cleaned_nlp_meta_dataset.csv")
SAMPLE_CLEANED_REVIEW_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, "cleaned/cleaned_reviews_dataset.csv")

# TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "graph/positives/train_data.pt")
# VAL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "graph/positives/val_data.pt")
# TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "graph/positives/test_data.pt")

SPLITED_GRAPH_SAVE_PATH =  os.path.join(DATA_DIR, "processed/splited_graph")
TRAIN_DATA_PATH = os.path.join(SPLITED_GRAPH_SAVE_PATH, "train_data.pt")
VAL_DATA_PATH = os.path.join(SPLITED_GRAPH_SAVE_PATH, "val_data.pt")
TEST_DATA_PATH = os.path.join(SPLITED_GRAPH_SAVE_PATH, "test_data.pt")

SAMPLE_DATA_SIZE = 10000
DATA_CHUNK_SIZE = 1000

TRAIN_DATA_SPLIT = 0.8
VAL_DATA_SPLIT = 0.1
TEST_DATA_SPLIT = 0.1

# --------------------------------------------------------------
# Model Settings
# --------------------------------------------------------------


SAVED_MODEL_PATH = os.path.join(PROJECT_ROOT, "artificats/")
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



# --------------------------------------------------------------
# Logging and Checkpointing
# --------------------------------------------------------------



# --------------------------------------------------------------
# Hardware Settings
# --------------------------------------------------------------



# --------------------------------------------------------------
# Random Seed and Reproducibility
# --------------------------------------------------------------



# --------------------------------------------------------------
# Hyperparameter Optimization (Optional)
# --------------------------------------------------------------


# --------------------------------------------------------------
# Logger
# --------------------------------------------------------------
SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SESSION_LOG_DIR = os.path.join("logs", SESSION_TIMESTAMP)


# --------------------------------------------------------------
# Miscellaneous
# --------------------------------------------------------------


