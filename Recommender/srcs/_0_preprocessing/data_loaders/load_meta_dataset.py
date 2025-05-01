import os
import sys
from typing import Optional, List

import pandas as pd

from logger import get_module_logger
from settings import (
    META_DATA_PATH,
    SAMPLE_DATA_SIZE,
    DATA_CHUNK_SIZE
)

# -----------------------------
# 📁 Setup
# -----------------------------
logger = get_module_logger("load_metadata_dataset")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# -----------------------------
# 🚀 Loader Function
# -----------------------------
def load_metadata_dataset(
    path: str = META_DATA_PATH,
    asins_to_keep: Optional[List[str]] = None,
    chunk_size: int = DATA_CHUNK_SIZE,
    max_records: int = SAMPLE_DATA_SIZE
) -> pd.DataFrame:
    """
    Loads and optionally filters the Amazon product metadata JSON dataset.

    Args:
        path (str): Path to the metadata JSONL file.
        asins_to_keep (Optional[List[str]]): List of parent_asins to retain.
        chunk_size (int): Number of rows to read per chunk.
        max_records (int): Maximum number of records to load.

    Returns:
        pd.DataFrame: Loaded metadata dataframe.
    """

    logger.info("=" * 80)
    logger.info("📦 LOADING METADATA DATASET".center(80))
    logger.info("=" * 80)
    logger.info(f"📂 Path          : {path}")
    logger.info(f"🔢 Max Records   : {max_records:,}")
    logger.info(f"📦 Chunk Size    : {chunk_size:,}")
    logger.info(f"🎯 Filter Active : {'Yes' if asins_to_keep else 'No'}")

    records_loaded = 0
    filtered_out = 0
    chunks = []

    try:
        logger.info("\n⏳ Loading Progress:\n")
        progress_bar_template = "[{bar:<20}] {percent:>3.0f}% | {current:,}/{total:,}"

        for idx, chunk in enumerate(pd.read_json(path, lines=True, chunksize=chunk_size)):
            original_len = len(chunk)

            if asins_to_keep:
                chunk = chunk[chunk['parent_asin'].isin(asins_to_keep)]
                filtered_out += original_len - len(chunk)

            chunks.append(chunk)
            records_loaded += len(chunk)

            if idx % 5 == 0 or records_loaded >= max_records:
                percent = min(records_loaded / max_records, 1.0)
                bar = "█" * int(percent * 20)
                logger.debug(progress_bar_template.format(
                    bar=bar,
                    percent=percent * 100,
                    current=records_loaded,
                    total=max_records
                ))

            if records_loaded >= max_records:
                break

        df = pd.concat(chunks, ignore_index=True)

        # Summary
        logger.info("\n" + "═" * 100)
        logger.info("✅ METADATA LOAD COMPLETE".center(100))
        logger.info("═" * 100)
        logger.info(f"📊 Total Records   : {len(df):,}")
        logger.info(f"🧹 Filtered Out    : {filtered_out:,}")
        logger.info(f"🧾 Columns         : {df.columns.tolist()}")

        # Sample Preview
        logger.info("\n🔍 Sample Record Preview:")
        for col, val in df.head(1).T.iterrows():
            sample_val = val.iloc[0]
            if isinstance(sample_val, str) and len(sample_val) > 100:
                sample_val = sample_val[:100] + "... [truncated]"
            logger.info(f"  • {col:<20}: {sample_val}")

        # Data Quality Report
        logger.info("\n🧪 Missing Values:")
        missing = df.isna().sum()
        total_missing = missing.sum()
        if total_missing == 0:
            logger.info("  ✅ No missing values found.")
        else:
            for col, count in missing.items():
                if count > 0:
                    logger.info(f"  • {col:<20}: {count:>6,} ({count/len(df):.1%})")

        return df

    except Exception as e:
        logger.error("\n❌ FAILED TO LOAD METADATA DATASET")
        logger.error(f"🚨 Error: {str(e)}")
        logger.error(f"📊 Records loaded: {records_loaded:,}/{max_records:,}")
        raise
