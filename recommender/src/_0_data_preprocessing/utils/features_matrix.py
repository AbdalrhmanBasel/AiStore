import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Any


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


# ==== Example Usage ====

if __name__ == "__main__":
    """
    Example usage of the feature extraction pipeline.

    Loads raw metadata, creates a feature matrix, engineers new features,
    normalizes the data, and saves the final feature matrix.
    """
    import os

    # Define paths
    RAW_META_PATH = "../../../data/raw/metadata_electronics_small.jsonl"
    SAVE_DIR = "../../../data/processed"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load raw metadata
    raw_meta_df = pd.read_json(RAW_META_PATH, lines=True)

    # Step 1: Create feature matrix
    feature_matrix = create_feature_matrix(raw_meta_df)

    # Step 2: Engineer additional features
    feature_matrix = feature_engineering(feature_matrix)

    # Step 3: Normalize features
    feature_matrix = normalize_features(feature_matrix)

    # Save the final feature matrix
    FEATURE_MATRIX_PATH = os.path.join(SAVE_DIR, "meta_feature_matrix.csv")
    feature_matrix.to_csv(FEATURE_MATRIX_PATH, index=False)

    # Display outputs
    print("Feature Matrix Sample:")
    print(feature_matrix.head(10))