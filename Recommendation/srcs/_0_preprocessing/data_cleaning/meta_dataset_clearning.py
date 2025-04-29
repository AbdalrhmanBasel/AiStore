from logger import get_module_logger
import pandas as pd
import numpy as np
import os
import sys

logger = get_module_logger("metadata_cleaning")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)

from settings import SAMPLE_CLEANED_META_DATA_PATH


def clean_metadata_dataset(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean metadata DataFrame for electronics products.
    
    Args:
        metadata_df: Raw metadata DataFrame.

    Returns:
        Cleaned DataFrame saved to SAMPLE_CLEANED_META_DATA_PATH and SAMPLE_CLEANED_META_DATA_JSONL_PATH.
    """
    logger.info("==== Starting Metadata Dataset Cleaning ====")
    original_count = len(metadata_df)

    # 1. Drop rows missing critical fields like 'title' and 'parent_asin'
    metadata_df = metadata_df.dropna(subset=["title", "parent_asin"])

    # 2. Convert 'average_rating', 'rating_number', 'price' to proper numeric types
    metadata_df["average_rating"] = pd.to_numeric(metadata_df["average_rating"], errors="coerce")
    metadata_df["rating_number"] = pd.to_numeric(metadata_df["rating_number"], errors="coerce").fillna(0).astype(int)
    metadata_df["price"] = pd.to_numeric(metadata_df["price"], errors="coerce").fillna(0)

    # 3. Clean text fields (remove leading/trailing spaces)
    text_fields = ["main_category", "title", "store"]
    for col in text_fields:
        if col in metadata_df.columns:
            metadata_df[col] = metadata_df[col].astype(str).str.strip()

    # 4. Validate structured fields (features, description, etc.)
    structured_fields = ["features", "description", "images", "videos", "categories", "details", "bought_together"]
    for field in structured_fields:
        if field in metadata_df.columns:
            metadata_df[field] = metadata_df[field].apply(
                lambda x: x if isinstance(x, (list, dict, type(np.nan))) else np.nan
            )

    # 5. Parse 'Date First Available' from 'details' if present (handling NaT for invalid dates)
    def extract_date(details):
        if isinstance(details, dict):
            date_str = details.get("Date First Available")
            try:
                return pd.to_datetime(date_str, errors='coerce')
            except Exception:
                return pd.NaT
        return pd.NaT

    # Apply date extraction and handle any invalid values
    metadata_df["date_first_available"] = metadata_df["details"].apply(extract_date)

    # 6. Remove future availability dates
    current_time = pd.Timestamp.now()
    future_dates = metadata_df["date_first_available"] > current_time
    if future_dates.any():
        logger.warning(f"Removing {future_dates.sum()} records with future 'Date First Available'")
        metadata_df = metadata_df[~future_dates]

    # 7. Additional Cleaning of 'price' to ensure there are no unexpected values (e.g., negative prices)
    metadata_df["price"] = metadata_df["price"].apply(lambda x: x if x >= 0 else np.nan)

    # 8. Drop rows with any remaining NaN values (e.g., from improperly parsed columns)
    metadata_df = metadata_df.dropna(subset=["price", "average_rating", "title", "parent_asin"])

    # 9. Save cleaned dataset (both CSV and JSONL formats)
    os.makedirs(os.path.dirname(SAMPLE_CLEANED_META_DATA_PATH), exist_ok=True)

    # Log head of cleaned metadata for inspection
    logger.info(f"Head of cleaned metadata:\n{metadata_df.head(1).to_string(index=False)}")

    # Save as CSV
    metadata_df.to_csv(SAMPLE_CLEANED_META_DATA_PATH, index=False)
    logger.info(f"Saved cleaned metadata to CSV: {SAMPLE_CLEANED_META_DATA_PATH}")

    # Save as JSONL
    metadata_df.to_json(SAMPLE_CLEANED_META_DATA_PATH, orient="records", lines=True, force_ascii=False)
    logger.info(f"Saved cleaned metadata to JSONL: {SAMPLE_CLEANED_META_DATA_PATH}")

    # Log results
    logger.info(f"Original records: {original_count:,}")
    logger.info(f"Final cleaned records: {len(metadata_df):,}")
    logger.info(f"Removed records: {original_count - len(metadata_df):,} ({100*(original_count - len(metadata_df))/original_count:.1f}%)")
    logger.info("==== Metadata Dataset Cleaning Completed Successfully ====")

    return metadata_df
