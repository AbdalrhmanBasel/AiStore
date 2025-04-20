import torch
from torch_geometric.data import Data
import pandas as pd

def build_graph(reviews_df: pd.DataFrame, meta_df: pd.DataFrame):
    """
    Builds a bipartite graph from reviews and metadata.
    Nodes: users and products
    Edges: reviews interactions (user <-> product)

    Args:
    - reviews_df: DataFrame containing reviews data.
    - meta_df: DataFrame containing product metadata.

    Returns:
    - A PyG Data object representing the graph.
    """
    print("Building graph...")

    # 1. Unique users and products in sampled data
    user_ids = reviews_df['user_id'].unique()
    product_ids = reviews_df['parent_asin'].unique()

    # 2. Index mapping
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    product_id_map = {pid: idx + len(user_ids) for idx, pid in enumerate(product_ids)}

    # 3. Build edges (from reviews)
    edge_list = []
    for _, row in reviews_df.iterrows():
        u_idx = user_id_map[row['user_id']]
        p_idx = product_id_map[row['parent_asin']]
        edge_list.append([u_idx, p_idx])
        edge_list.append([p_idx, u_idx])  # make it undirected

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 4. Node Features
    # -- Users: number of reviews
    user_features = reviews_df.groupby('user_id').size().reindex(user_ids).fillna(0).values
    user_features = torch.tensor(user_features, dtype=torch.float).view(-1, 1)

    # -- Products: average rating (from metadata)
    product_meta = meta_df.set_index('parent_asin')
    avg_ratings = [product_meta.loc[pid]['average_rating'] if pid in product_meta.index else 0 for pid in product_ids]
    product_features = torch.tensor(avg_ratings, dtype=torch.float).view(-1, 1)

    # Combine user and product features
    x = torch.cat([user_features, product_features], dim=0)

    # 5. Create Data object for PyTorch Geometric
    data = Data(x=x, edge_index=edge_index)

    print(f"Graph built with {x.shape[0]} nodes and {edge_index.shape[1]} edges.")
    return data



