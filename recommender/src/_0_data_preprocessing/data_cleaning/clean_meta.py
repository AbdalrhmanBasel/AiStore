import pandas as pd
import numpy as np
import re
import ast

import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)


from settings import INTERIM_DATA_DIR, META_DATA_PATH


def safe_literal_eval(x):
    """
    Safely evaluate a string representation of a Python literal (list/dict).
    Handles missing or invalid values gracefully.

    Args:
        x: Input value (string, list, dict, or NaN).

    Returns:
        Parsed object (list, dict) or default empty structure if invalid.
    """
    if isinstance(x, (list, dict)):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return [] if not isinstance(x, dict) else {}
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return x


def parse_price(x):
    """
    Parse price from various formats into a float.
    Handles strings with symbols, missing values, and numeric inputs.

    Args:
        x: Price input (string, int, float, or NaN).

    Returns:
        Float price or NaN if parsing fails.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    cleaned = re.sub(r'[^0-9.]', '', str(x))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame by filling with column medians.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    for col in [
        'average_rating', 'rating_number', 'num_features', 'description_length',
        'price', 'num_images', 'num_videos', 'num_categories', 'num_details'
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def clean_meta(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features from Amazon metadata DataFrame.

    Args:
        meta_df (pd.DataFrame): Raw metadata DataFrame.

    Returns:
        pd.DataFrame: Cleaned and feature-engineered metadata DataFrame.
    """
    df = meta_df.copy()

    # 1) Parse JSON-like columns (lists/dicts stored as strings)
    json_cols = ['categories', 'details', 'features', 'images', 'videos', 'bought_together']
    for col in json_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_literal_eval)
        else:
            # If missing, fill with empty lists/dicts
            default = {} if col == 'details' else []
            df[col] = [default] * len(df)

    # 2) Basic feature counts
    df['num_categories'] = df['categories'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_details'] = df['details'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
    df['num_features'] = df['features'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_images'] = df['images'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_videos'] = df['videos'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # 3) Textual feature: Description length
    df['description_length'] = df.get('description', pd.Series([''] * len(df))).fillna('').apply(
        lambda s: len(str(s).split()))

    # 4) Numeric parsing
    df['average_rating'] = pd.to_numeric(df.get('average_rating', 0), errors='coerce')
    df['rating_number'] = pd.to_numeric(df.get('rating_number', 0), errors='coerce')
    df['price'] = df.get('price', np.nan).apply(parse_price)

    # 5) Handle missing values
    df = handle_missing_values(df)

    return df


# ==== Example Usage ====

if __name__ == "__main__":
    """
    Example usage of the `clean_meta` function.

    Loads raw metadata, cleans it, and saves the cleaned version.
    """

    SAVE_DIR = INTERIM_DATA_DIR
    os.makedirs(SAVE_DIR, exist_ok=True)

    raw_meta_df = pd.read_json(META_DATA_PATH, lines=True)
    cleaned_meta = clean_meta(raw_meta_df)

    CLEANED_META_PATH = os.path.join(SAVE_DIR, "cleaned_meta.csv")
    cleaned_meta.to_csv(CLEANED_META_PATH, index=False)

    print("Cleaned Metadata Sample:")
    print(cleaned_meta.head(10))