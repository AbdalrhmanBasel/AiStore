import pandas as pd
from typing import List, Optional
import torch
import os, sys 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', '..'))
sys.path.append(project_root)


from settings import SAMPLE_DATA_SIZE, DATA_CHUNK_SIZE


def load_reviews_dataset(
    path: str,
    sample_size: int = 30_000,
    chunk_size: int = 10_000
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
    print(f"🔄 Sampling {sample_size} reviews from '{path}' in chunks of {chunk_size}...")

    sampled_reviews = []
    sample_count = 0

    for chunk in pd.read_csv(path, chunksize=chunk_size):
        remaining = sample_size - sample_count
        n_samples = min(remaining, len(chunk))
        
        sampled_chunk = chunk.sample(n=n_samples, random_state=42)
        sampled_reviews.append(sampled_chunk)
        sample_count += n_samples

        if sample_count >= sample_size:
            break

    print(f"✅ Successfully loaded {sample_count} reviews.")
    return pd.concat(sampled_reviews, ignore_index=True)


def load_metadata_dataset(
    path: str,
    asins_to_keep: Optional[List[str]] = None,
    chunk_size: int = 10_000,
    max_records: int = 100_000
) -> pd.DataFrame:
    """
    Loads product metadata from a large JSONL file using chunking,
    and filters to only include ASINs of interest.
    reviews_path = "../../../data/raw/reviews_electronics_small.csv"
    metadata_path = "../../../data/raw/metadata_electronics_small.json"
    Returns:
        pd.DataFrame: Filtered product metadata.
    """
    print(f"🔄 Loading metadata from '{path}' in chunks of {chunk_size}...")

    chunks = []
    records_loaded = 0

    for chunk in pd.read_json(path, lines=True, chunksize=chunk_size):
        if asins_to_keep is not None:
            chunk = chunk[chunk['parent_asin'].isin(asins_to_keep)]

        chunks.append(chunk)
        records_loaded += len(chunk)

        if records_loaded >= max_records:
            break

    metadata_df = pd.concat(chunks, ignore_index=True)
    print(f"✅ Loaded {len(metadata_df)} filtered metadata records.")
    return metadata_df


def load_data(reviews_path, metadata_path, df_sample_size=SAMPLE_DATA_SIZE, df_chunk_size=DATA_CHUNK_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and filters a sample of electronics reviews and corresponding product metadata.

    Args:
        reviews_path (str): Path to the reviews dataset.
        metadata_path (str): Path to the metadata dataset.
        df_sample_size (int, optional): The sample size to load from the reviews dataset. Defaults to SAMPLE_DATA_SIZE.
        df_chunk_size (int, optional): The chunk size to use when loading the dataset. Defaults to DATA_CHUNK_SIZE.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing two DataFrames (reviews_df, metadata_df).
    """
    
    reviews_df = load_reviews_dataset(path=reviews_path, sample_size=df_sample_size, chunk_size=df_chunk_size)
    asins = reviews_df['parent_asin'].unique().tolist()
    metadata_df = load_metadata_dataset(path=metadata_path, asins_to_keep=asins)

    return reviews_df, metadata_df



def save_graph(data, path="graph_data.pt"):
    """
    Save the PyTorch Geometric Data object to disk.

    Args:
        data (torch_geometric.data.Data): The graph data to be saved.
        path (str): The file path where the graph data will be saved (default: "graph_data.pt").
    """
    torch.save(data, path)
    print(f"[INFO] Graph saved to: {path}")


def load_graph(path="graph_data.pt", map_location='cpu'):
    """
    Load the PyTorch Geometric Data object from disk.

    Args:
        path (str): Path to the saved .pt file (default: "graph_data.pt").
        map_location (str): The device to load the data onto (default: 'cpu').

    Returns:
        data (torch_geometric.data.Data): The loaded graph data object.
    """
    if os.path.exists(path):
        data = torch.load(path, map_location=map_location)
        print(f"[INFO] Graph loaded from: {path}")
        print(f"[INFO] Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_node_features}")
        return data
    else:
        raise FileNotFoundError(f"No graph data found at {path}")