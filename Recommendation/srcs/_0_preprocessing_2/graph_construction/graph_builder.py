from logger import get_module_logger
import os
import sys
import torch
from typing import Tuple, Dict
import pandas as pd
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm

logger = get_module_logger("graph_builder")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)

from settings import (
    SAMPLE_CLEANED_META_DATA_PATH,
    SAMPLE_CLEANED_REVIEW_DATA_PATH,
    FEATURES_MATRIX_PATH,
    DATA_CHUNK_SIZE
)

def graph_builder():
    """Main function to build the recommendation graph with comprehensive logging"""
    logger.info("🔄 Starting graph building process")
    
    try:
        # Load data with validation
        logger.info("Loading cleaned data files")
        meta_df = pd.read_csv(SAMPLE_CLEANED_META_DATA_PATH)
        feature_matrix = pd.read_csv(FEATURES_MATRIX_PATH)
        reviews_df = pd.read_csv(SAMPLE_CLEANED_REVIEW_DATA_PATH)
        
        logger.debug(f"Meta data shape: {meta_df.shape}")
        logger.debug(f"Feature matrix shape: {feature_matrix.shape}")
        logger.debug(f"Reviews data shape: {reviews_df.shape}")
        
        # Validate required columns
        required_meta_cols = ['parent_asin']
        required_review_cols = ['user_id', 'parent_asin', 'rating']
        
        for col in required_meta_cols:
            if col not in meta_df.columns:
                raise ValueError(f"Missing required column in meta data: {col}")
        
        for col in required_review_cols:
            if col not in reviews_df.columns:
                raise ValueError(f"Missing required column in reviews data: {col}")
        
        # Construct graph
        graph = construct_graph(meta_df, feature_matrix, reviews_df)
        logger.info("✅ Graph building completed successfully")
        return graph
        
    except Exception as e:
        logger.error("❌ Graph building failed", exc_info=True)
        raise

def encode_ids(meta_df: pd.DataFrame, reviews_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Create ID mappings with detailed logging"""
    logger.info("Generating ID mappings")
    
    try:
        product_ids = meta_df['parent_asin'].unique()
        user_ids = reviews_df['user_id'].unique()
        
        logger.debug(f"Found {len(product_ids)} unique products")
        logger.debug(f"Found {len(user_ids)} unique users")
        
        product2idx = {pid: idx for idx, pid in enumerate(tqdm(product_ids, desc="Mapping products"))}
        user2idx = {uid: idx + len(product2idx) for idx, uid in enumerate(tqdm(user_ids, desc="Mapping users"))}
        
        logger.debug(f"Product mapping covers {len(product2idx)} items")
        logger.debug(f"User mapping covers {len(user2idx)} users")
        
        return product2idx, user2idx
        
    except Exception as e:
        logger.error("Failed to generate ID mappings", exc_info=True)
        raise

def construct_graph(
    meta_df: pd.DataFrame,
    meta_features_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    product_col: str = 'parent_asin',
    user_col: str = 'user_id',
    rating_col: str = 'rating',
    chunk_size: int = DATA_CHUNK_SIZE
) -> Data:
    """Build graph with detailed progress tracking"""
    logger.info("Constructing graph dataset")
    
    try:
        # 1. ID Mapping
        logger.debug("Creating ID mappings")
        product2idx, user2idx = encode_ids(meta_df, reviews_df)
        
        # 2. Edge Construction
        logger.info("Building edges")
        edges = []
        edge_attrs = []
        total_edges = 0
        
        chunks = np.array_split(reviews_df, len(reviews_df)//chunk_size + 1)
        logger.debug(f"Processing {len(chunks)} chunks of max size {chunk_size}")
        
        for chunk in tqdm(chunks, desc="Processing edges"):
            for _, row in chunk.iterrows():
                if row[product_col] in product2idx and row[user_col] in user2idx:
                    edges.append([user2idx[row[user_col]], product2idx[row[product_col]]])
                    edge_attrs.append([float(row[rating_col])])
                    total_edges += 1
        
        logger.debug(f"Created {total_edges} valid edges")
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # 3. Node Features
        logger.info("Creating node features")
        logger.debug(f"Original feature matrix columns: {meta_features_df.columns.tolist()}")
        
        # Ensure parent_asin exists in feature matrix
        if product_col not in meta_features_df.columns:
            logger.warning(f"Adding {product_col} to feature matrix from meta data")
            meta_features_df[product_col] = meta_df[product_col]
        
        features_df = meta_features_df.set_index(product_col).loc[list(product2idx.keys())]
        logger.debug(f"Feature matrix after filtering: {features_df.shape}")
        
        x_products = torch.tensor(features_df.values, dtype=torch.float)
        x_users = torch.zeros((len(user2idx), x_products.size(1)), dtype=torch.float)
        x = torch.cat([x_products, x_users], dim=0)
        
        # 4. Create Data Object
        logger.info("Finalizing graph object")
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_products=len(product2idx),
            num_users=len(user2idx)
        )
        
        logger.info(f"Graph constructed with {data.num_nodes} nodes and {data.num_edges} edges")
        logger.debug(f"Node features shape: {x.shape}")
        logger.debug(f"Edge index shape: {edge_index.shape}")
        logger.debug(f"Edge attributes shape: {edge_attr.shape}")
        
        return data
        
    except Exception as e:
        logger.error("Graph construction failed", exc_info=True)
        raise