"""
settings.py

Configuration file for the StoreX GNN-based Recommendation System.
Defines global paths, constants, hyperparameters, and logging structure.
"""

import os
from datetime import datetime

# --------------------------------------------------------------
# Project Metadata
# --------------------------------------------------------------
PROJECT_NAME = "StoreX GNN Recommendation System"
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Logging
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGGING_DIR = os.path.join(PROJECT_ROOT, "logs")
TENSORBOARD_LOG_DIR = os.path.join(LOGGING_DIR, "tensorboard")
SESSION_LOG_DIR = os.path.join(LOGGING_DIR, SESSION_TIMESTAMP)

# --------------------------------------------------------------
# Dataset Directories & Paths
# --------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")

# Full Dataset Paths
META_DATA_PATH = os.path.join(RAW_DATA_DIR, "meta_Electronics.jsonl")
REVIEW_DATA_PATH = os.path.join(RAW_DATA_DIR, "Electronics_2.csv")

# Sample Dataset Paths
SAMPLE_DATA_DIR = os.path.join(INTERIM_DATA_DIR, "samples")
SAMPLE_META_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, "raw/metadata_small.jsonl")
SAMPLE_REVIEW_DATA_PATH = os.path.join(SAMPLE_DATA_DIR, "raw/reviews_small.csv")
SAMPLE_CLEANED_META_PATH = os.path.join(SAMPLE_DATA_DIR, "raw/cleaned_meta_dataset.jsonl")
SAMPLE_CLEANED_REVIEW_PATH = os.path.join(SAMPLE_DATA_DIR, "cleaned/cleaned_reviews_dataset.csv")
SAMPLE_NLP_CLEANED_META_PATH = os.path.join(SAMPLE_DATA_DIR, "cleaned/cleaned_nlp_meta_dataset.csv")

# Graph Storage Paths
GRAPH_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "graph")
GRAPH_SPLIT_DIR = os.path.join(PROCESSED_DATA_DIR, "splited_graph")
FEATURE_MATRIX_PATH = os.path.join(INTERIM_DATA_DIR, "graph/feature_matrices/feature_matrix.csv")

EDGE_INDEX_PATH = os.path.join(GRAPH_DATA_DIR, "edge_index.pt")
FEATURES_PATH = os.path.join(GRAPH_DATA_DIR, "features.pt")
LABELS_PATH = os.path.join(GRAPH_DATA_DIR, "labels.pt")
NODE_MAPPING_PATH = os.path.join(GRAPH_DATA_DIR, "mappings.pkl")

# Train/Val/Test Graph Splits
TRAIN_DATA_PATH = os.path.join(GRAPH_SPLIT_DIR, "train_data.pt")
VAL_DATA_PATH = os.path.join(GRAPH_SPLIT_DIR, "val_data.pt")
TEST_DATA_PATH = os.path.join(GRAPH_SPLIT_DIR, "test_data.pt")

# --------------------------------------------------------------
# Data Loading & Sampling
# --------------------------------------------------------------
SAMPLE_DATA_SIZE = 10000     # Number of samples to load from full dataset
DATA_CHUNK_SIZE = 1000      # Number of rows to read at a time

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# --------------------------------------------------------------
# Model Configuration
# --------------------------------------------------------------
MODEL_NAME = "GraphSAGE"

# Architecture
IN_CHANNELS = 4
HIDDEN_CHANNELS = 64
OUT_CHANNELS = 32
NUM_LAYERS = 2
DROPOUT = 0.5

# Model Saving
# SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, "artifacts/saved_models")
SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, "artifacts/saved_models/GraphSAGE_model.pth")
EMBEDDINGS_SAVE_PATH = os.path.join(PROJECT_ROOT, "artifacts/node_embeddings.pt")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "artifacts/saved_checkpoints")
TRAINED_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, f"{MODEL_NAME}_final.pth")
VISUALIZATION_PATH = os.path.join(PROJECT_ROOT, "artifacts/visualizations")
# --------------------------------------------------------------
# Training Hyperparameters
# --------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 5  # For early stopping
L2_REGULARIZATION = 1e-5
MOMENTUM = 0.9
GRADIENT_CLIP = 5.0  # Set to None to disable

# Scheduler
SCHEDULER = "StepLR"        # Options: StepLR, ExponentialLR, etc.
LR_STEP_SIZE = 10
LR_GAMMA = 0.7

# --------------------------------------------------------------
# Evaluation Metrics (to be defined in training/evaluation modules)
# --------------------------------------------------------------
# Define evaluation metrics like accuracy, precision, recall, F1, ROC AUC etc.

# --------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------
# Consider setting random seeds in your training script
# Example: torch.manual_seed(SEED)
# SEED = 42
DEVICE = "cpu"
# --------------------------------------------------------------
# Hyperparameter Tuning (Optional)
# --------------------------------------------------------------
# Use Optuna or similar frameworks in a separate script

