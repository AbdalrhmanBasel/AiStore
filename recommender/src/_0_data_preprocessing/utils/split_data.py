import torch
import numpy as np
import random
from torch_geometric.data import Data



def split_edges(graph_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    num_edges = graph_data.num_edges
    train_size = int(num_edges * train_ratio)
    val_size = int(num_edges * val_ratio)
    test_size = num_edges - train_size - val_size

    # Randomly shuffle edges for splitting
    perm = torch.randperm(num_edges)
    train_edges = perm[:train_size]
    val_edges = perm[train_size:train_size+val_size]
    test_edges = perm[train_size+val_size:]

    # Create new edge index tensors for train, val, and test
    train_data = graph_data.edge_index[:, train_edges]
    val_data = graph_data.edge_index[:, val_edges]
    test_data = graph_data.edge_index[:, test_edges]

    return train_data, val_data, test_data


def negative_sampling(graph_data, num_negative_samples):
    """
    Generates negative edges (non-existent edges) for link prediction.
    """
    edge_index = graph_data.edge_index
    num_nodes = graph_data.num_nodes
    
    negative_edges = []
    
    # Generate negative samples
    for _ in range(num_negative_samples):
        # Randomly sample pairs of nodes that are not connected by an edge
        while True:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            
            # Ensure u != v and there is no edge between u and v
            if u != v and not torch.any((edge_index[0] == u) & (edge_index[1] == v)) and not torch.any((edge_index[0] == v) & (edge_index[1] == u)):
                negative_edges.append([u, v])
                break
    
    # Convert negative edges to tensor
    negative_edges = torch.tensor(negative_edges, dtype=torch.long).t().contiguous()
    
    return negative_edges


def create_data_for_link_prediction(graph_data, num_negative_samples=10000):
    """
    Prepare data for link prediction task by adding negative samples.
    """
    # Generate negative samples
    negative_edges = negative_sampling(graph_data, num_negative_samples)
    
    # Combine positive and negative edges
    positive_edges = graph_data.edge_index
    all_edges = torch.cat([positive_edges, negative_edges], dim=1)
    
    # Labels: 1 for positive edges (real edges), 0 for negative edges
    edge_labels = torch.cat([torch.ones(positive_edges.size(1), dtype=torch.float), 
                             torch.zeros(negative_edges.size(1), dtype=torch.float)], dim=0)
    
    # Return modified graph data with additional info for link prediction
    data = Data(x=graph_data.x, edge_index=all_edges, edge_attr=edge_labels)
    
    return data


