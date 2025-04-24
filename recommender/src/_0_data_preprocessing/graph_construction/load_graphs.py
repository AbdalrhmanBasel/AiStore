import pandas as pd
from typing import List, Optional
import torch
import os
import sys
from colorama import Fore, Style, init
from src._0_data_preprocessing.graph_construction.report_graph import report_graph_details

# Initialize colorama for cross-platform compatibility
init()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)

from settings import SAMPLE_DATA_SIZE, DATA_CHUNK_SIZE, GRAPH_SAVE_PATH, REVIEW_DATA_PATH, META_DATA_PATH


def load_reviews_dataset(
    path: str,
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
    print(f"{Fore.YELLOW}🔄 Sampling {sample_size} reviews from '{path}' in chunks of {chunk_size}...{Style.RESET_ALL}")

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

        print(f"{Fore.GREEN}✅ Successfully loaded {sample_count} reviews.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to load reviews dataset: {e}{Style.RESET_ALL}")
        raise

    return pd.concat(sampled_reviews, ignore_index=True)


def load_metadata_dataset(
    path: str,
    asins_to_keep: Optional[List[str]] = None,
    chunk_size: int = DATA_CHUNK_SIZE,
    max_records: int = 100_000
) -> pd.DataFrame:
    """
    Loads product metadata from a large JSONL file using chunking,
    and filters to only include ASINs of interest.

    Args:
        path (str): Path to the metadata JSONL file.
        asins_to_keep (Optional[List[str]]): List of ASINs to filter by.
        chunk_size (int): Number of rows to process in each chunk.
        max_records (int): Maximum number of records to load.

    Returns:
        pd.DataFrame: Filtered product metadata.
    """
    print(f"{Fore.YELLOW}🔄 Loading metadata from '{path}' in chunks of {chunk_size}...{Style.RESET_ALL}")

    chunks = []
    records_loaded = 0

    try:
        for chunk in pd.read_json(path, lines=True, chunksize=chunk_size):
            if asins_to_keep is not None:
                chunk = chunk[chunk['parent_asin'].isin(asins_to_keep)]

            chunks.append(chunk)
            records_loaded += len(chunk)

            if records_loaded >= max_records:
                break

        metadata_df = pd.concat(chunks, ignore_index=True)
        print(f"{Fore.GREEN}✅ Loaded {len(metadata_df)} filtered metadata records.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to load metadata dataset: {e}{Style.RESET_ALL}")
        raise

    return metadata_df


def load_data(
    reviews_path: str = REVIEW_DATA_PATH,
    metadata_path: str = META_DATA_PATH,
    df_sample_size: int = SAMPLE_DATA_SIZE,
    df_chunk_size: int = DATA_CHUNK_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and filters a sample of electronics reviews and corresponding product metadata.

    Args:
        reviews_path (str): Path to the reviews dataset.
        metadata_path (str): Path to the metadata dataset.
        df_sample_size (int): The sample size to load from the reviews dataset.
        df_chunk_size (int): The chunk size to use when loading the dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing two DataFrames (reviews_df, metadata_df).
    """
    print(f"{Fore.CYAN}[INFO] Loading reviews and metadata datasets...{Style.RESET_ALL}")

    try:
        # Load reviews dataset
        reviews_df = load_reviews_dataset(path=reviews_path, sample_size=df_sample_size, chunk_size=df_chunk_size)

        # Extract unique ASINs from reviews
        asins = reviews_df['parent_asin'].unique().tolist()

        # Load metadata dataset filtered by ASINs
        metadata_df = load_metadata_dataset(path=metadata_path, asins_to_keep=asins)

        print(f"{Fore.GREEN}✅ Successfully loaded reviews and metadata datasets.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to load data: {e}{Style.RESET_ALL}")
        raise

    return reviews_df, metadata_df


def save_graph(data, path: str = GRAPH_SAVE_PATH):
    """
    Save the PyTorch Geometric Data object to disk.

    Args:
        data (torch_geometric.data.Data): The graph data to be saved.
        path (str): The file path where the graph data will be saved.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(data, path)
        print(f"{Fore.GREEN}[SUCCESS] Graph saved to: {path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to save graph: {e}{Style.RESET_ALL}")
        raise


def load_graph(path: str = GRAPH_SAVE_PATH, map_location: str = 'cpu'):
    """
    Load the PyTorch Geometric Data object from disk and report its details.

    Args:
        path (str): Path to the saved .pt file.
        map_location (str): The device to load the data onto.

    Returns:
        torch_geometric.data.Data: The loaded graph data object.
    """
    try:
        # Check if the file exists
        if os.path.exists(path):
            print(f"{Fore.GREEN}[INFO] Loading graph from: {path}{Style.RESET_ALL}")
            data = torch.load(path, map_location=map_location)

            # Report basic graph details
            print(f"{Fore.BLUE}[INFO] Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_node_features}{Style.RESET_ALL}")

            # Report detailed graph attributes
            report_graph_details(data)

            return data
        else:
            raise FileNotFoundError(f"{Fore.RED}[ERROR] No graph data found at {path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to load graph: {e}{Style.RESET_ALL}")
        raise