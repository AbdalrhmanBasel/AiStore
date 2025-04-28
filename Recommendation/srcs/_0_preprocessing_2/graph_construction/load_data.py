from logger import get_module_logger
from typing import List, Optional

import os
import sys
import torch
from datetime import datetime
from typing import Tuple
import pandas as pd
from torch_geometric.data import Data

from srcs._0_preprocessing_2.data_cleaning.clean_meta import clean_meta
from srcs._0_preprocessing_2.data_cleaning.clean_reviews import clean_reviews
from srcs._0_preprocessing_2.graph_construction.report_graph import report_graph_details
from srcs._0_preprocessing_2.graph_construction.graph_builder import construct_graph

logger = get_module_logger("load_data")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")

sys.path.append(PROJECT_ROOT)
from settings import (
    META_DATA_PATH,
    REVIEW_DATA_PATH,
    COMPLETE_GRAPH_SAVE_PATH,
    SAMPLE_CLEANED_META_DATA_PATH,
    SAMPLE_CLEANED_REVIEW_DATA_PATH,
    SAMPLE_DATA_SIZE, 
    DATA_CHUNK_SIZE
)

def load_and_clean_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean raw data with comprehensive logging"""
    logger.info("Starting data loading and cleaning pipeline")
    
    try:
        # Load data
        logger.debug(f"Loading reviews from {REVIEW_DATA_PATH}")
        reviews_df, meta_df = load_data()
        logger.info(f"Loaded raw data - Reviews: {reviews_df.shape}, Meta: {meta_df.shape}")

        # Clean data
        logger.debug("Cleaning datasets")
        cleaned_reviews = clean_reviews(reviews_df)
        cleaned_meta = clean_meta(meta_df)
        logger.info(f"Cleaned data - Reviews: {cleaned_reviews.shape}, Meta: {cleaned_meta.shape}")

        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(SAMPLE_CLEANED_REVIEW_DATA_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(SAMPLE_CLEANED_META_DATA_PATH), exist_ok=True)

        # Save samples
        cleaned_reviews.to_csv(SAMPLE_CLEANED_REVIEW_DATA_PATH, index=False)
        cleaned_meta.to_csv(SAMPLE_CLEANED_META_DATA_PATH, index=False)
        logger.debug(f"Saved cleaned samples to {SAMPLE_CLEANED_REVIEW_DATA_PATH}")

        return cleaned_reviews, cleaned_meta

    except Exception as e:
        logger.error(f"Data loading/cleaning failed: {str(e)}", exc_info=True)
        raise


def build_graph(cleaned_meta: pd.DataFrame, feature_matrix: pd.DataFrame, reviews_df: pd.DataFrame) -> Data:
    """Graph construction with resource monitoring"""
    logger.info("Building product-user interaction graph")
    
    try:
        start_time = datetime.now()
        
        graph = construct_graph(
            meta_df=cleaned_meta,
            meta_features_df=feature_matrix,
            reviews_df=reviews_df
        )
        
        # Log graph statistics
        report_graph_details(graph)
        logger.info(f"Graph built - Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
        
        # Save and verify
        torch.save(graph, COMPLETE_GRAPH_SAVE_PATH)
        loaded = torch.load(COMPLETE_GRAPH_SAVE_PATH)
        assert loaded.num_nodes == graph.num_nodes, "Graph save/load verification failed"
        
        logger.info(f"Graph construction completed in {(datetime.now()-start_time).total_seconds():.2f}s")
        return graph
        
    except Exception as e:
        logger.critical("Graph construction failed", exc_info=True)
        raise


def load_graph(path: str = COMPLETE_GRAPH_SAVE_PATH, map_location: str = 'cpu'):
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
            logger.info(f"Loading graph from: {path}")
            data = torch.load(path, map_location=map_location)

            # Report basic graph details
            logger.info(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_node_features}")

            # Report detailed graph attributes
            report_graph_details(data)

            return data
        else:
            raise FileNotFoundError(f"No graph data found at {path}")
    except Exception as e:
        logger.error(f"Failed to load graph: {e}")
        raise


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
    logger.info("Loading reviews and metadata datasets...")

    try:
        # Load reviews dataset
        reviews_df = load_reviews_dataset(path=reviews_path, sample_size=df_sample_size, chunk_size=df_chunk_size)
        print(reviews_df.head())

        # Extract unique ASINs from reviews
        asins = reviews_df['parent_asin'].unique().tolist()
        
        # Load metadata dataset filtered by ASINs
        metadata_df = load_metadata_dataset(path=metadata_path, asins_to_keep=asins)
        print(metadata_df.head())

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    return reviews_df, metadata_df


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

        logger.info(f"Successfully loaded {sample_count} reviews.")
    except Exception as e:
        logger.error(f"Failed to load reviews dataset: {e}")
        raise

    return pd.concat(sampled_reviews, ignore_index=True)


def load_metadata_dataset(
    path: str = META_DATA_PATH,
    asins_to_keep: Optional[List[str]] = None,
    chunk_size: int = DATA_CHUNK_SIZE,
    max_records: int = SAMPLE_DATA_SIZE
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
    logger.info(f"Loading metadata from '{path}' in chunks of {chunk_size}...")

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
        logger.info(f"Loaded {len(metadata_df)} filtered metadata records.")
    except Exception as e:
        logger.error(f"Failed to load metadata dataset: {e}")
        raise

    return metadata_df