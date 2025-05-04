import os
import torch

# --------------------------------------------------------------
# General Project Settings
# --------------------------------------------------------------
PROJECT_NAME = "StoreXGNN"
ROOT_DIR = os.getcwd()  

# --------------------------------------------------------------
# Dataset Directories & Paths
# --------------------------------------------------------------

# Main directories for storing raw and processed data
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')  
CLEANED_DATA_DIR = os.path.join(DATA_DIR, 'cleaned')  
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')  

# File paths for raw data
REVIEWS_PATH = os.path.join(RAW_DATA_DIR, 'reviews.csv')
METADATA_PATH = os.path.join(RAW_DATA_DIR, 'meta.jsonl')

FILTERED_METADATA_PATH_JSON = os.path.join(RAW_DATA_DIR, 'filtered_meta.jsonl')
FILTERED_METADATA_PATH_CSV = os.path.join(RAW_DATA_DIR, 'filtered_meta.csv')
FILTERED_METADATA_PATH_PARQUET = os.path.join(RAW_DATA_DIR, 'filtered_meta.parquet')

CLEANED_REVIEWS_PATH_CSV = os.path.join(CLEANED_DATA_DIR, 'cleaned_reviews.csv')
CLEANED_REVIEWS_PATH_PARQUET = os.path.join(CLEANED_DATA_DIR, 'cleaned_reviews.parquet')
CLEANED_METADATA_PATH_CSV = os.path.join(CLEANED_DATA_DIR, 'cleaned_meta.csv')
CLEANED_METADATA_PATH_PARQUET = os.path.join(CLEANED_DATA_DIR, 'cleaned_meta.parquet')

# Path to the processed graph data
FULL_GRAPH_PATH = os.path.join(PROCESSED_DATA_DIR, 'graph.pt')  
TRAIN_GRAPH_PATH = os.path.join(PROCESSED_DATA_DIR, 'graph_splits/graph_train.pt')  
VAL_GRAPH_PATH = os.path.join(PROCESSED_DATA_DIR, 'graph_splits/graph_val.pt')  
TEST_GRAPH_PATH = os.path.join(PROCESSED_DATA_DIR, 'graph_splits/graph_test.pt')  

# --------------------------------------------------------------
# Model & Artifact Directories
# --------------------------------------------------------------

# Main directory for saving model artifacts like embeddings and checkpoints
ARTIFACTS_DIR = os.path.join(ROOT_DIR, 'artifacts')
MODEL_DIR = os.path.join(ARTIFACTS_DIR, 'models')  
LOGS_DIR = os.path.join(ARTIFACTS_DIR, 'logs')  
IMAGES_DIR = os.path.join(ARTIFACTS_DIR, 'images')  

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "graphsage_model.pt")
# Specific paths for model-related files
EMBEDDING_PATH = os.path.join(MODEL_DIR, 'embeddings')
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'checkpoints') 

# --------------------------------------------------------------
# Hyperparameters for Training
# --------------------------------------------------------------

# General training hyperparameters
LEARNING_RATE = 0.001 
BATCH_SIZE = 32  
EPOCHS = 100
HIDDEN_CHANNELS = 64  
NUM_LAYERS = 2
DROPOUT= 0.5
IN_CHANNELS = 5  # Input feature dimensions (user/item features)

NUM_USERS = 10000  # Number of unique users in the dataset
NUM_ITEMS = 5000  # Number of unique items in the dataset
NUM_NEGATIVE_SAMPLES = 5  # Number of negative samples per positive sample for training

# --------------------------------------------------------------
# GraphSAGE Specific Hyperparameters
# --------------------------------------------------------------

# Specific hyperparameters for the GraphSAGE model
GRAPH_SAGE_LAYERS = 2  # Number of layers in the GraphSAGE model
AGGREGATOR = "mean"  # Aggregation method (mean, pooling, etc.)

# --------------------------------------------------------------
# Sampling Configuration
# --------------------------------------------------------------

# Settings related to negative sampling during training
NEGATIVE_SAMPLING_RATIO = 1  # Ratio of negative samples to positive samples
SAMPLER_BATCH_SIZE = 1024  # Batch size for negative sampling during training

# --------------------------------------------------------------
# Evaluation Settings
# --------------------------------------------------------------

TOP_K = 10  
EVAL_METRICS = ['hit_at_k', 'ndcg_at_k'] 

# --------------------------------------------------------------
# CUDA Configuration (GPU/CPU)
# --------------------------------------------------------------

USE_CUDA = True 
DEVICE = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu" 

# --------------------------------------------------------------
# Logging & Checkpointing
# --------------------------------------------------------------

LOG_EVERY_N_BATCHES = 100  # Log training progress every N batches
MODEL_SAVE_INTERVAL = 5  # Save model checkpoint every N epochs

# --------------------------------------------------------------
# Random Seed for Reproducibility
# --------------------------------------------------------------

SEED = 42  

# --------------------------------------------------------------
# Ensure All Necessary Directories are Created
# --------------------------------------------------------------

def create_directories():
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
        MODEL_DIR, LOGS_DIR, EMBEDDING_PATH, 
        CHECKPOINT_PATH, IMAGES_DIR 
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")


# create_directories()
