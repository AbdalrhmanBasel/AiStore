import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split

def load_data():
    # 1. Load the graph data (edges and node features)
    # This is just a placeholder. Replace it with your actual data loading logic
    edge_list = pd.read_csv('../../processed/edge_list.csv')  # Assumed edge list in CSV format (source, target)
    node_features = pd.read_csv('../../processed/node_features.csv')  # Assumed node features in CSV format (node_id, feature_1, feature_2, ...)
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor(edge_list.values.T, dtype=torch.long)  # edge_index should be shape [2, num_edges]
    x = torch.tensor(node_features.values[:, 1:], dtype=torch.float)  # node features, skipping the node_id column
    
    # 2. Prepare positive (train_pos) and negative (train_neg) edges for training
    # Assuming that positive edges are the ones in the edge list:
    train_pos = edge_index.T  # train_pos: shape [num_train_edges, 2]

    # Sample negative edges:
    # Generate random pairs of nodes and filter out the existing edges (non-existing edges).
    num_nodes = x.size(0)
    all_possible_edges = torch.combinations(torch.arange(num_nodes), r=2).T
    negative_mask = ~torch.isin(all_possible_edges.T, train_pos)
    neg_edges = all_possible_edges[:, negative_mask].T

    # Select a fixed number of negative edges (e.g., same number as positive edges)
    num_neg_samples = train_pos.size(0)
    neg_edges = neg_edges[:num_neg_samples]

    train_neg = neg_edges

    # 3. Create the Data object for the graph
    data = Data(x=x, edge_index=edge_index, train_pos=train_pos, train_neg=train_neg)

    return data
