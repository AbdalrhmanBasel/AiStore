from logger import get_module_logger
import os
import sys
import pandas as pd

logger = get_module_logger("reviews_cleaning")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)


def clean_review_dataset(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean reviews DataFrame with comprehensive validation.

    Args:
        reviews_df: Raw reviews DataFrame containing columns:
                   ['user_id', 'parent_asin', 'rating', 'timestamp']

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logger.info("==== 🧹 Starting Reviews Dataset Cleaning ====")
    original_count = len(reviews_df)
    logger.info(f"\n\n🔍 Initial dataset sample:\n{reviews_df.head(5).to_string(index=False)}")

    # 1. Basic Cleaning
    reviews_df = reviews_df.dropna(subset=['rating', 'user_id', 'parent_asin'])
    reviews_df = reviews_df[reviews_df['rating'].between(1, 5)]
    logger.info(f"\n\n🛠️ After basic cleaning sample:\n{reviews_df.head(5).to_string(index=False)}")

    # 2. Convert timestamp (assuming Unix ms)
    reviews_df['timestamp'] = pd.to_datetime(reviews_df['timestamp'], unit='ms')
    logger.info(f"\n\n🕰️ After timestamp conversion sample:\n{reviews_df.head(5).to_string(index=False)}")

    # 3. Remove duplicates (keep most recent)
    reviews_df = reviews_df.sort_values('timestamp', ascending=False)\
                           .drop_duplicates(['user_id', 'parent_asin'])
    logger.info(f"\n\n📚 After duplicates removal sample:\n{reviews_df.head(5).to_string(index=False)}")

    # 4. Remove future timestamps
    current_time = pd.Timestamp.now()
    future_dates = reviews_df['timestamp'] > current_time
    if future_dates.any():
        logger.warning(f"⚠️ Removing {future_dates.sum()} records with future timestamps")
        reviews_df = reviews_df[~future_dates]

    logger.info(f"\n\n🧹 After future timestamps cleaning sample:\n{reviews_df.head(5).to_string(index=False)}")

    # Log results
    logger.info(f"✅ Original records: {original_count:,}")
    logger.info(f"✅ Final cleaned records: {len(reviews_df):,}")
    logger.info(f"✅ Removed records: {original_count - len(reviews_df):,} "
                f"({100*(original_count - len(reviews_df))/original_count:.1f}%)")
    logger.info("==== 🏁 Reviews Dataset Cleaning Completed Successfully ====")

    return reviews_df
