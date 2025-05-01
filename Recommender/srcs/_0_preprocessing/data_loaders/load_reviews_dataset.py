from logger import get_module_logger
import os
import sys
import pandas as pd

logger = get_module_logger("load_review_data")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)

from settings import (
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
    Efficiently loads a random sample of reviews from a large CSV file using chunking.

    Parameters:
        path (str): Path to the reviews CSV file.
        sample_size (int): Total number of reviews to sample.
        chunk_size (int): Number of rows to process per chunk.

    Returns:
        pd.DataFrame: A DataFrame containing the sampled reviews.
    """
    logger.info("=" * 80)
    logger.info("📝 REVIEW DATA LOADING".center(80))
    logger.info("=" * 80)
    logger.info(f"📄 Source: {path}")
    logger.info(f"🎯 Target Sample Size: {sample_size:,}")
    logger.info(f"🔹 Chunk Size: {chunk_size:,}")

    sampled_reviews = []
    sample_count = 0
    total_rows_read = 0
    try:
        for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
            chunk_len = len(chunk)
            total_rows_read += chunk_len

            remaining = sample_size - sample_count
            if remaining <= 0:
                break

            n_samples = min(remaining, chunk_len)
            sampled_chunk = chunk.sample(n=n_samples, random_state=42)
            sampled_reviews.append(sampled_chunk)
            sample_count += len(sampled_chunk)

            logger.debug(f"✅ Processed Chunk {i+1:>3}: {chunk_len:,} rows | Sampled: {n_samples:,} | Total Sampled: {sample_count:,}")

            if sample_count >= sample_size:
                break

        final_df = pd.concat(sampled_reviews, ignore_index=True)

        # Final summary
        logger.info("═" * 100)
        logger.info("✅ REVIEW DATA LOAD COMPLETE".center(100))
        logger.info("═" * 100)
        logger.info(f"{'• Total Sampled:':<20} {sample_count:>58,}")
        logger.info(f"{'• Rows Read:':<20} {total_rows_read:>58,}")
        logger.info(f"{'• Columns:':<20} {str(list(final_df.columns)):>58}")

        # Sample preview
        logger.info("\n✨ Sample Records:")
        logger.info("\n" + final_df.head(5).to_string(index=False))

        return final_df

    except FileNotFoundError:
        logger.error("❌ File not found. Please check the path.")
        raise

    except pd.errors.EmptyDataError:
        logger.error("❌ The CSV file is empty.")
        raise

    except Exception as e:
        logger.error("❌ An unexpected error occurred during review data loading.")
        logger.error(f"Error: {str(e)}")
        raise
