from logger import get_module_logger
import pandas as pd
import numpy as np
import os
import sys

logger = get_module_logger("metadata_cleaning")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)

def clean_metadata_dataset(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean metadata DataFrame for electronics products.

    Args:
        metadata_df: Raw metadata DataFrame.

    Returns:
        Cleaned metadata DataFrame (not saved to disk).
    """
    logger.info("==== Starting Metadata Dataset Cleaning ====")
    original_count = len(metadata_df)

    # 1. Drop rows missing critical fields
    metadata_df = metadata_df.dropna(subset=["title", "parent_asin"])

    # 2. Convert rating and price fields to numeric
    metadata_df["average_rating"] = pd.to_numeric(metadata_df["average_rating"], errors="coerce")
    metadata_df["rating_number"] = pd.to_numeric(metadata_df["rating_number"], errors="coerce").fillna(0).astype(int)
    metadata_df["price"] = pd.to_numeric(metadata_df["price"], errors="coerce").fillna(0)

    # 3. Clean text fields
    for col in ["main_category", "title", "store"]:
        if col in metadata_df.columns:
            metadata_df[col] = metadata_df[col].astype(str).str.strip()

    # 4. Validate structured fields
    for field in ["features", "description", "images", "videos", "categories", "details", "bought_together"]:
        if field in metadata_df.columns:
            metadata_df[field] = metadata_df[field].apply(
                lambda x: x if isinstance(x, (list, dict, type(np.nan))) else np.nan
            )

    # 5. Extract 'Date First Available' from 'details'
    def extract_date(details):
        if isinstance(details, dict):
            return pd.to_datetime(details.get("Date First Available"), errors="coerce")
        return pd.NaT

    metadata_df["date_first_available"] = metadata_df["details"].apply(extract_date)

    # 6. Remove entries with future dates
    now = pd.Timestamp.now()
    future_mask = metadata_df["date_first_available"] > now
    if future_mask.any():
        logger.warning(f"Removing {future_mask.sum()} records with future availability dates")
        metadata_df = metadata_df[~future_mask]

    # 7. Ensure prices are non-negative
    metadata_df["price"] = metadata_df["price"].apply(lambda x: x if x >= 0 else np.nan)

    # 8. Drop rows with NaNs in key fields
    metadata_df = metadata_df.dropna(subset=["price", "average_rating", "title", "parent_asin"])

    # 9. Logging
    logger.info(f"Head of cleaned metadata:\n{metadata_df.head(1).to_string(index=False)}")
    logger.info(f"Original records: {original_count:,}")
    logger.info(f"Cleaned records: {len(metadata_df):,}")
    logger.info(f"Records removed: {original_count - len(metadata_df):,} ({100 * (original_count - len(metadata_df)) / original_count:.1f}%)")
    logger.info("==== Metadata Dataset Cleaning Completed ====")

    return metadata_df



