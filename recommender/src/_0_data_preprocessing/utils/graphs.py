import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional


def encode_ids(
    meta_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    product_col: str = "parent_asin",
    user_col: str = "user_id"
) -> Tuple[Dict[str,int], Dict[str,int]]:
    """
    Generate mapping dictionaries for product and user IDs to unique integers.
    """
    product_ids = sorted(meta_df[product_col].unique())
    user_ids = sorted(reviews_df[user_col].unique())
    product2idx = {pid: idx for idx, pid in enumerate(product_ids)}
    user2idx = {uid: idx + len(product2idx) for idx, uid in enumerate(user_ids)}
    return product2idx, user2idx


def construct_graph(
    meta_df: pd.DataFrame,
    meta_features_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    product_col: str = 'parent_asin',
    user_col: str = 'user_id',
    rating_col: str = 'rating',
    label_col: Optional[str] = None
) -> Data:
    """
    Constructs a bipartite user-product graph for GNNs.

    Args:
        meta_df: Cleaned metadata (must include product_col).
        meta_features_df: DataFrame of numeric features (no index on product).
        reviews_df: Cleaned reviews (must include product_col, user_col, rating_col).
        product_col: Column name for product IDs.
        user_col: Column name for user IDs.
        rating_col: Column name for edge attributes (ratings).
        label_col: Optional product label column in meta_df.

    Returns:
        PyG Data object with x, edge_index, edge_attr, and optional y.
    """
    # Encode IDs
    product2idx, user2idx = encode_ids(meta_df, reviews_df, product_col, user_col)
    product_ids = list(product2idx.keys())

    # Build edge list and attributes
    edges = []
    edge_attrs = []
    for _, row in reviews_df.iterrows():
        pid = row[product_col]
        uid = row[user_col]
        if pid in product2idx and uid in user2idx:
            edges.append([user2idx[uid], product2idx[pid]])
            edge_attrs.append([row.get(rating_col, 0.0)])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # Align feature matrix with products
    features_df = meta_features_df.copy()
    if product_col not in features_df.columns:
        features_df[product_col] = meta_df[product_col].values
    features_df = features_df.set_index(product_col)
    features_df = features_df.loc[product_ids]
    x_products = torch.tensor(features_df.values, dtype=torch.float)

    # Create user features as zeros
    x_users = torch.zeros((len(user2idx), x_products.size(1)), dtype=torch.float)

    # Concatenate
    x = torch.cat([x_products, x_users], dim=0)

    # Optional labels
    y = None
    if label_col and label_col in meta_df.columns:
        labels = meta_df.set_index(product_col).loc[product_ids][label_col].values
        y = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = y
    return data


def save_graph(data: Data, path: str = "graph_data.pt") -> None:
    torch.save(data, path)
    print(f"[INFO] Graph saved to: {path}")


def load_graph(path: str = "graph_data.pt", map_location: str = "cpu") -> Data:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] No file found at {path}")
    data = torch.load(path, map_location=map_location)
    print(f"[INFO] Graph loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    return data


def save_graph_numpy(data: Data, dir_path: str = "graph_np") -> None:
    os.makedirs(dir_path, exist_ok=True)
    np.save(os.path.join(dir_path, "edge_index.npy"), data.edge_index.numpy())
    np.save(os.path.join(dir_path, "x.npy"), data.x.numpy())
    np.save(os.path.join(dir_path, "edge_attr.npy"), data.edge_attr.numpy())
    print(f"[INFO] Graph saved as NumPy arrays to: {dir_path}")


def load_graph_numpy(dir_path: str = "graph_np") -> Data:
    edge_index = torch.from_numpy(np.load(os.path.join(dir_path, "edge_index.npy")))
    x = torch.from_numpy(np.load(os.path.join(dir_path, "x.npy")))
    edge_attr = torch.from_numpy(np.load(os.path.join(dir_path, "edge_attr.npy")))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(f"[INFO] Graph loaded from NumPy: {x.shape[0]} nodes, {edge_index.shape[1]} edges")
    return data

def create_labeled_graph(pos_edges, neg_samples, num_nodes):
    """
    Combines positive and negative edges into a single graph with labels.

    Args:
        pos_edges (torch.Tensor): Positive edge indices (shape [2, num_pos_edges]).
        neg_samples (torch_geometric.data.Data): Negative samples as a Data object.
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        torch_geometric.data.Data: Combined graph with labeled edges.
    """
    # Extract negative edge indices from the Data object
    neg_edge_index = neg_samples.edge_index

    # Combine positive and negative edges
    all_edges = torch.cat([pos_edges, neg_edge_index], dim=1)

    # Create labels: 1 for positive edges, 0 for negative edges
    pos_labels = torch.ones(pos_edges.shape[1], dtype=torch.float)
    neg_labels = torch.zeros(neg_edge_index.shape[1], dtype=torch.float)
    all_labels = torch.cat([pos_labels, neg_labels], dim=0)

    # Return the combined graph as a Data object
    return torch_geometric.data.Data(
        x=torch.eye(num_nodes),  # Use identity matrix as node features (if none provided)
        edge_index=all_edges,
        y=all_labels,
        num_nodes=num_nodes
    )