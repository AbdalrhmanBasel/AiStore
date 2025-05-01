from logger import get_module_logger
import os
import sys

import pandas as pd
from typing import List, Optional

logger = get_module_logger("load_data")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)

from settings import (
    META_DATA_PATH,
    REVIEW_DATA_PATH,
    SAMPLE_DATA_SIZE, 
    DATA_CHUNK_SIZE
)

def load_reviews_dataset(
    path: str = REVIEW_DATA_PATH,
    sample_size: int = SAMPLE_DATA_SIZE,
    chunk_size: int = DATA_CHUNK_SIZE
) -> pd.DataFrame:
    """
    Loads a random sample of reviews from a large CSV file using chunking.

    Args:
        path (str): Path to the reviews CSV file.
        sample_size (int): Total number of reviews to sample.
        chunk_size (int): Number of rows to process in each chunk.

    Returns:
        pd.DataFrame: A DataFrame containing the sampled reviews.
    """
    logger.info(f"Sampling {sample_size} reviews from '{path}' in chunks of {chunk_size}...")

    sampled_reviews = []
    sample_count = 0

    try:
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            remaining = sample_size - sample_count
            n_samples = min(remaining, len(chunk))

            sampled_chunk = chunk.sample(n=n_samples, random_state=42)
            sampled_reviews.append(sampled_chunk)
            sample_count += n_samples

            if sample_count >= sample_size:
                break

        final_df = pd.concat(sampled_reviews, ignore_index=True)
        
        logger.info("Sample reviews dataset preview:")
        logger.info("\n" + str(final_df.head(5)))
        
        logger.info(f"Successfully loaded {sample_count} reviews.")
        return final_df
        
    except Exception as e:
        logger.error(f"Failed to load reviews dataset: {e}")
        raise


def load_metadata_dataset(
    path: str = META_DATA_PATH,
    asins_to_keep: Optional[List[str]] = None,
    chunk_size: int = DATA_CHUNK_SIZE,
    max_records: int = SAMPLE_DATA_SIZE
) -> pd.DataFrame:
    """
    Loads product metadata with beautifully formatted logging output
    """
    # Header with border
    logger.info("="*80)
    logger.info(f"📦 Loading Metadata Dataset".center(80))
    logger.info("="*80)
    logger.info(f"🔍 Source: {path}")
    logger.info(f"📏 Max Records: {max_records:,} | Chunk Size: {chunk_size:,}")
    logger.info(f"🎯 ASIN Filter: {'Active' if asins_to_keep else 'Inactive'}")
    
    chunks = []
    records_loaded = 0
    filtered_out = 0

    try:
        # Progress bar simulation
        logger.info("\n\n⏳ Loading progress:")
        progress_template = "[{bar}] {percent:.0f}% | {n_fmt}/{total_fmt} records"
        
        for chunk_idx, chunk in enumerate(pd.read_json(path, lines=True, chunksize=chunk_size)):
            original_size = len(chunk)
            
            if asins_to_keep:
                chunk = chunk[chunk['parent_asin'].isin(asins_to_keep)]
                filtered_out += original_size - len(chunk)
            
            chunks.append(chunk)
            records_loaded += len(chunk)
            
            # Update progress every 5 chunks
            if chunk_idx % 5 == 0:
                logger.debug(progress_template.format(
                    bar="▋" * int(records_loaded/max_records*20),
                    percent=records_loaded/max_records*100,
                    n_fmt=records_loaded,
                    total_fmt=max_records
                ))
            
            if records_loaded >= max_records:
                break

        metadata_df = pd.concat(chunks, ignore_index=True)
        
        # Final summary with box
        logger.info("═"*100)
        logger.info("✅ METADATA LOAD COMPLETE".center(100))
        logger.info("═"*100)
        logger.info(f"{'• Total Loaded:':<20} {len(metadata_df):>58,}")
        logger.info(f"{'• Filtered Out:':<20} {filtered_out:>58,}")
        logger.info(f"{'• Columns:':<20} {str(list(metadata_df.columns)):>58}")
        
        # Pretty-printed sample
        logger.info("\n\n✨ Sample Record Preview:")
        sample = metadata_df.head(1).T  # Transpose for vertical display
        for col, val in sample.items():
            logger.info(f"  {col}:")
            for i, (k, v) in enumerate(val.items()):
                if isinstance(v, str) and len(v) > 100:
                    v = v[:100] + "... [truncated]"
                logger.info(f"    {k}: {v}")
        
        # Data quality report
        logger.info("\n\n🔍 Data Quality Check:")
        missing = metadata_df.isna().sum()
        logger.info(f"  Missing Values ({missing.sum()} total):")
        for col, count in missing.items():
            if count > 0:
                logger.info(f"    {col:<20}: {count:>6,} ({count/len(metadata_df):.1%})")
        
        return metadata_df
        
    except Exception as e:
        logger.error("\n❌ LOAD FAILED")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Progress: {records_loaded:,}/{max_records:,} records")
        raise