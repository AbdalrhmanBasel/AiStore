import os
import sys
import re
from datetime import datetime
from typing import Dict, Tuple, List
import pickle
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler

from logger import get_module_logger

from settings import (
    EDGE_INDEX_PATH,
    FEATURES_PATH,
    LABELS_PATH,
    MAPPING_GRAPH_PATH,
    COMPLETE_GRAPH_SAVE_PATH
)

# ─── Setup ──────────────────────────────────────────────────────────────────────

logger = get_module_logger("data_mapping")
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "../../../"))
sys.path.append(PROJECT_ROOT)

# ─── Utility Functions ───────────────────────────────────────────────────────────

def parse_statistic(series: pd.Series, method: str = 'median', fallback: float = 0.0) -> float:
    """Compute the median or mean of a series, fallback if NaN."""
    value = series.median(skipna=True) if method == 'median' else series.mean(skipna=True)
    return float(value) if not np.isnan(value) else fallback

def extract_temporal_features(timestamps: pd.Series) -> pd.DataFrame:
    """Extract year, month, weekday, and days since epoch from timestamps."""
    dt = pd.to_datetime(timestamps, errors='coerce').fillna(pd.Timestamp(1970, 1, 1))
    return pd.DataFrame({
        "year": dt.dt.year.astype(float),
        "month": dt.dt.month.astype(float),
        "day_of_week": dt.dt.dayofweek.astype(float),
        "days_since_epoch": (dt - pd.Timestamp(1970, 1, 1)).dt.days.astype(float)
    })

def remove_constant_columns(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove columns with zero variance."""
    variances = np.nanvar(arr, axis=0)
    keep_mask = variances > 0
    return arr[:, keep_mask], keep_mask

def standardize_features(arr: np.ndarray) -> np.ndarray:
    """Standardize features after removing constants."""
    arr, keep_mask = remove_constant_columns(arr)
    scaler = StandardScaler()
    return scaler.fit_transform(arr), keep_mask

def log_tensor_info(name: str, tensor: torch.Tensor, sample_size: int = 5):
    """Log tensor information."""
    logger.info(f"{name} → shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}")
    logger.debug(f"{name} sample: {tensor[:sample_size].tolist()}")

# ─── Core Mapping Function ───────────────────────────────────────────────────────

def map_users_products_to_pyg(
    users: pd.Series,
    products: pd.Series,
    metadata: pd.DataFrame,
    interactions: pd.DataFrame
) -> Tuple[Data, Dict[str, int]]:
    """Construct PyG Data object from user-product interaction data."""
    logger.info("==== Starting Graph Construction ====")

    # ─── ID Mappings ────────────────────────────────────────────────────────────
    user_ids = users.unique()
    product_ids = products.unique()
    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    product2idx = {pid: idx + len(user_ids) for idx, pid in enumerate(product_ids)}
    id2idx = {**user2idx, **product2idx}

    logger.info(f"Users: {len(user_ids)}, Products: {len(product_ids)}")

    # ─── Edge Index ─────────────────────────────────────────────────────────────
    src = users.map(user2idx).to_numpy(dtype=int)
    dst = products.map(product2idx).to_numpy(dtype=int)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_index = to_undirected(edge_index)
    log_tensor_info("edge_index", edge_index)

    # ─── Product Features ───────────────────────────────────────────────────────
    metadata = metadata.set_index('parent_asin')
    med_price = parse_statistic(metadata.get('price', pd.Series([])), method='median')
    mean_rating = parse_statistic(metadata.get('average_rating', pd.Series([])), method='mean')

    product_features = []
    for pid in product_ids:
        record = metadata.loc[pid] if pid in metadata.index else None
        price = record.get('price', med_price) if record is not None else med_price
        avg_rating = record.get('average_rating', mean_rating) if record is not None else mean_rating
        rating_count = record.get('rating_number', 0) if record is not None else 0
        category_count = len(record.get('categories', [])) if record is not None and isinstance(record.get('categories'), list) else 0
        product_features.append([float(price), float(avg_rating), float(rating_count), float(category_count)])

    prod_arr = np.array(product_features, dtype=float)
    log_tensor_info("raw_product_features", torch.tensor(prod_arr, dtype=torch.float))

    # ─── User Features ──────────────────────────────────────────────────────────
    now = pd.Timestamp.now()
    user_stats = interactions.groupby('user_id').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count'),
        first_interaction=('timestamp', 'min'),
        last_interaction=('timestamp', 'max')
    ).reset_index()

    user_features = []
    for uid in user_ids:
        record = user_stats[user_stats['user_id'] == uid]
        if not record.empty:
            record = record.iloc[0]
            days_since_first = (now - pd.to_datetime(record['first_interaction'], errors='coerce')).days
            days_since_last = (now - pd.to_datetime(record['last_interaction'], errors='coerce')).days
            user_features.append([
                float(record['avg_rating']),
                float(record['rating_count']),
                float(days_since_first),
                float(days_since_last)
            ])
        else:
            user_features.append([mean_rating, 0., 0., 0.])

    user_arr = np.array(user_features, dtype=float)
    log_tensor_info("raw_user_features", torch.tensor(user_arr, dtype=torch.float))

    # ─── Edge Attributes ────────────────────────────────────────────────────────
    temporal_feats = extract_temporal_features(interactions['timestamp'])
    temporal_feats['rating_normalized'] = interactions['rating'].fillna(0.) / 5.0
    edge_attr = torch.tensor(temporal_feats.to_numpy(dtype=float), dtype=torch.float)
    log_tensor_info("edge_attr", edge_attr)

    # ─── Feature Normalization ──────────────────────────────────────────────────
    user_arr_norm, _ = standardize_features(user_arr)
    prod_arr_norm, _ = standardize_features(prod_arr)

    log_tensor_info("normalized_user_features", torch.tensor(user_arr_norm, dtype=torch.float))
    log_tensor_info("normalized_product_features", torch.tensor(prod_arr_norm, dtype=torch.float))

    # ─── Assemble Node Features ─────────────────────────────────────────────────
    x = torch.cat([
        torch.tensor(user_arr_norm, dtype=torch.float),
        torch.tensor(prod_arr_norm, dtype=torch.float)
    ], dim=0)
    log_tensor_info("node_features (x)", x)

    # ─── Build Graph ────────────────────────────────────────────────────────────
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_users=len(user_ids),
        num_products=len(product_ids)
    )
    logger.info(f"Constructed PyG graph: {graph}")
    logger.info("==== Graph Construction Completed ====")

    # ─── Save Graph ────────────────────────────────────────────────────────────
    save_graph_data(graph, id2idx)

    return graph, id2idx



def save_graph_data(graph: Data, id2idx: dict):
    """
    Save PyG Data object and mappings directly into the output directory.
    """
    # Paths
    edge_index_path = EDGE_INDEX_PATH
    features_path = FEATURES_PATH
    labels_path = LABELS_PATH
    mappings_path = MAPPING_GRAPH_PATH

    # Save graph components
    torch.save(graph.edge_index, edge_index_path)
    logger.info(f"[save_graph_data] Saved edge_index tensor: {edge_index_path} | Shape: {graph.edge_index.shape}")

    torch.save(graph.x, features_path)
    logger.info(f"[save_graph_data] Saved node features tensor: {features_path} | Shape: {graph.x.shape}")

    torch.save(graph.edge_attr, labels_path)
    logger.info(f"[save_graph_data] Saved edge attributes tensor: {labels_path} | Shape: {graph.edge_attr.shape}")

    # Save mappings
    with open(mappings_path, 'wb') as f:
        pickle.dump(id2idx, f)
    logger.info(f"[save_graph_data] Saved ID to index mapping: {mappings_path} | Total mappings: {len(id2idx)}")

    logger.info(f"[save_graph_data] ✅ All graph components successfully saved to {COMPLETE_GRAPH_SAVE_PATH}")
