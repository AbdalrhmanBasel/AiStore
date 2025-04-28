from logger import get_module_logger
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Any

import os
import sys

logger = get_module_logger("load_data")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)

from settings import FEATURES_MATRIX_PATH

def generate_feature_matrix(cleaned_meta: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering pipeline with validation"""
    logger.info("Generating feature matrix")
    
    try:
        # Create output directory if not exists
        os.makedirs(os.path.dirname(FEATURES_MATRIX_PATH), exist_ok=True)
        
        # Rest of the existing code
        base_features = create_feature_matrix(cleaned_meta)
        engineered = feature_engineering(base_features)
        normalized = normalize_features(engineered)
        
        # Validation and saving
        if normalized.isnull().any().any():
            logger.warning("Null values detected in feature matrix")
            normalized = normalized.fillna(0)
        
        normalized.to_csv(FEATURES_MATRIX_PATH, index=False)
        logger.info(f"Feature matrix saved to {FEATURES_MATRIX_PATH}")
        return normalized
        
    except Exception as e:
        logger.error("Feature matrix generation failed", exc_info=True)
        raise




# def generate_feature_matrix(cleaned_meta: pd.DataFrame) -> pd.DataFrame:
#     """Feature engineering pipeline with validation"""
#     logger.info("Generating feature matrix")
    
#     try:
#         # Feature creation
#         base_features = create_feature_matrix(cleaned_meta)
#         engineered = feature_engineering(base_features)
#         normalized = normalize_features(engineered)
        
#         # Validation
#         if normalized.isnull().any().any():
#             logger.warning("Null values detected in feature matrix")
#             normalized = normalized.fillna(0)
        
#         normalized.to_csv(FEATURES_MATRIX_PATH, index=False)
#         logger.info(f"Feature matrix saved to {FEATURES_MATRIX_PATH} (shape: {normalized.shape})")
        
#         return normalized
        
#     except Exception as e:
#         logger.error("Feature matrix generation failed", exc_info=True)
#         raise



def _safe_len(x: Any) -> int:
    """
    Safely compute the length of a list-like object.

    Args:
        x (Any): Input value (list-like or other).

    Returns:
        int: Length of the list-like object, or 0 if not applicable.
    """
    return len(x) if isinstance(x, list) else 0


def _extract_features(row: pd.Series) -> dict:
    """
    Extract structured features from a raw metadata row.

    Args:
        row (pd.Series): A single row from the metadata DataFrame.

    Returns:
        dict: Dictionary of extracted features for the row.
    """
    try:
        return {
            'average_rating': float(row.get('average_rating', 0.0)),
            'rating_number': int(row.get('rating_number', 0)),
            'num_features': _safe_len(row.get('features')),
            'description_length': len(row['description'][0])
                if isinstance(row.get('description'), list) and row['description'] else 0,
            'price': float(row['price']) if row.get('price') not in [None, '', np.nan] else 0.0,
            'num_images': _safe_len(row.get('images')),
            'num_videos': _safe_len(row.get('videos')),
            'num_categories': _safe_len(row.get('categories')),
        }
    except Exception as e:
        print(f"[WARN] Failed to extract features from row: {e}")
        return {
            'average_rating': 0.0,
            'rating_number': 0,
            'num_features': 0,
            'description_length': 0,
            'price': 0.0,
            'num_images': 0,
            'num_videos': 0,
            'num_categories': 0,
        }


def create_feature_matrix(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a numeric feature matrix from the product metadata DataFrame.

    Args:
        meta_df (pd.DataFrame): Raw metadata DataFrame.

    Returns:
        pd.DataFrame: DataFrame with extracted numeric features.
    """
    features = [_extract_features(row) for _, row in meta_df.iterrows()]
    return pd.DataFrame(features)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features from the base features.

    Args:
        df (pd.DataFrame): Feature matrix.

    Returns:
        pd.DataFrame: Feature matrix with additional engineered features.
    """
    # Price per rating
    df['price_per_rating'] = df['price'] / (df['rating_number'] + 1e-5)  # Avoid division by zero

    # Feature-to-rating ratio
    df['feature_to_rating_ratio'] = df['num_features'] / (df['rating_number'] + 1e-5)

    # Handle extreme values or NaNs post-engineering
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def normalize_features(df: pd.DataFrame, additional_cols: list = None) -> pd.DataFrame:
    """
    Normalize numerical features using Min-Max scaling.

    Args:
        df (pd.DataFrame): Feature DataFrame.
        additional_cols (list, optional): Any extra numeric columns to normalize.

    Returns:
        pd.DataFrame: Scaled DataFrame with values in [0, 1].
    """
    # Default numeric columns
    numeric_cols = [
        'average_rating', 'rating_number', 'num_features', 'description_length',
        'price', 'num_images', 'num_videos', 'num_categories',
        'price_per_rating', 'feature_to_rating_ratio'
    ]

    # Add additional columns if provided
    if additional_cols:
        numeric_cols.extend(additional_cols)

    # Filter only existing numeric columns
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    # Apply Min-Max scaling
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df