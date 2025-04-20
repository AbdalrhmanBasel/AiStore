import pandas as pd
from typing import List, Optional

def load_reviews_sample(path: str, sample_size: int = 10_000, chunk_size: int = 10_000) -> pd.DataFrame:
    """
    Loads a random sample of reviews from a CSV file in chunks.

    Args:
    - path: Path to the reviews CSV file.
    - sample_size: Total number of reviews to sample.
    - chunk_size: Size of each chunk to be processed in memory.

    Returns:
    - A DataFrame containing a random sample of reviews.
    """
    print(f"Sampling {sample_size} reviews from {path} in chunks of {chunk_size} ...")
    
    sampled_reviews = []
    chunk_iter = pd.read_csv(path, chunksize=chunk_size)
    sample_count = 0
    
    for chunk in chunk_iter:
        chunk_sample = chunk.sample(n=min(sample_size - sample_count, len(chunk)), random_state=42)
        sampled_reviews.append(chunk_sample)
        sample_count += len(chunk_sample)
        
        if sample_count >= sample_size:
            break

    return pd.concat(sampled_reviews, ignore_index=True)



def load_meta(path: str, asins_to_keep: Optional[List[str]] = None, chunk_size: int = 10000, max_records: int = 100000) -> pd.DataFrame:
    """
    Loads product metadata in chunks and filters relevant products.

    Args:
    - path: Path to the JSONL file.
    - asins_to_keep: List of parent ASINs to filter relevant products.
    - chunk_size: Size of each chunk to load.
    - max_records: Maximum number of records to load (limit for safety).

    Returns:
    - DataFrame with filtered metadata.
    """
    print(f"Loading metadata from {path} using pandas in chunks...")

    chunks = []
    records_loaded = 0
    
    # Read the file in chunks
    for chunk in pd.read_json(path, lines=True, chunksize=chunk_size):
        # Filter based on asins_to_keep if provided
        if asins_to_keep is not None:
            chunk = chunk[chunk['parent_asin'].isin(asins_to_keep)]

        # Append the chunk to the list of chunks
        chunks.append(chunk)
        records_loaded += len(chunk)

        # Stop if we've loaded the maximum number of records
        if records_loaded >= max_records:
            break
    
    # Concatenate all chunks into a single DataFrame
    df = pd.concat(chunks, ignore_index=True)

    print(f"✅ Loaded {len(df)} metadata entries after filtering.")
    return df

def clean_reviews(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean reviews DataFrame by removing unnecessary rows.
    """
    # Ensure we only keep reviews with a rating and user_id
    reviews_df = reviews_df.dropna(subset=['rating', 'user_id'])
    
    # Filter reviews that have valid ratings (1 to 5)
    reviews_df = reviews_df[reviews_df['rating'].isin([1, 2, 3, 4, 5])]
    
    return reviews_df


