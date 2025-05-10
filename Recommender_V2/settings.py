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

DATA_DIR = os.path.join(ROOT_DIR, 'data')
ARTIFACTS_DIR = os.path.join(ROOT_DIR, 'artifacts')

# Raw Data
REVIEWS_DATA_PATH = os.path.join(DATA_DIR, 'reviews.csv')
META_DATA_PATH = os.path.join(DATA_DIR, 'meta.csv')

# Graph Data
GRAPH_FULL_PATH = os.path.join(ARTIFACTS_DIR, 'data/graphs/graph_full.pt')
GRAPH_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, 'data/graphs/graph_train.pt')
GRAPH_VAL_PATH = os.path.join(ARTIFACTS_DIR, 'data/graphs/graph_val.pt')
GRAPH_TEST_PATH = os.path.join(ARTIFACTS_DIR, 'data/graphs/graph_test.pt')

ENCODER_USER = os.path.join(ARTIFACTS_DIR, 'encoders/user_encoder.pkl')
ENCODER_ITEM = os.path.join(ARTIFACTS_DIR, 'encoders/item_encoder.pkl')

ENCODER_CATEGORY = os.path.join(ARTIFACTS_DIR, 'encoders/category_encoder.pkl')
ENCODER_BRAND = os.path.join(ARTIFACTS_DIR, 'encoders/brand_encoder.pkl')
ENCODER_COLOR = os.path.join(ARTIFACTS_DIR, 'encoders/color_encoder.pkl')
ENCODER_SCALER = os.path.join(ARTIFACTS_DIR, 'scalers/feature_scaler.pkl')

MAPPING_USER_T0_ID_PATH = os.path.join(ARTIFACTS_DIR, 'mappings/user_to_id_mapping.json')
MAPPING_ITEM_ASIN_TO_ID_PATH = os.path.join(ARTIFACTS_DIR, 'mappings/item_asin_to_id_mapping.json')
MAPPIN_ID_TO_ASIN_ITEM_PATH = os.path.join(ARTIFACTS_DIR, 'mappings/id_to_item_asin_mapping.json')

ITEM_FEATURE_PREPROCESSOR = os.path.join(ARTIFACTS_DIR, 'preprocessor/item_feature_preprocessor.pkl')
# --------------------------------------------------------------
# Model & Artifact Directories
# --------------------------------------------------------------

# Main directory for saving model artifacts like embeddings and checkpoints

GNN_MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "models/HeroGraphSAGE_model.pth")
PREDICTOR_MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "models/Predictor_model.pth")
CLEANED_META_DATA_PATH = os.path.join(ARTIFACTS_DIR, 'meta_df_clean.pkl')

IMAGES_DIR = os.path.join(ARTIFACTS_DIR, 'images')

# --------------------------------------------------------------
# Hyperparameters for Training
# --------------------------------------------------------------



# --------------------------------------------------------------
# GraphSAGE Specific Hyperparameters
# --------------------------------------------------------------



# --------------------------------------------------------------
# Sampling Configuration
# --------------------------------------------------------------


# --------------------------------------------------------------
# Evaluation Settings
# --------------------------------------------------------------


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
