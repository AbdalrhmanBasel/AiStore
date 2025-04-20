import torch
import random
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

class LinkPredictionDataset:
    def __init__(self, edge_index, num_nodes, split_ratio=0.8):
        """
        edge_index: Tensor of shape [2, num_edges]
        num_nodes: Total number of nodes
        """
        self.num_nodes = num_nodes
        self.edge_index = edge_index

        # Step 1: Shuffle and split edges into train/test
        self.train_edges, self.test_edges = self.split_edges(split_ratio)

        # Step 2: Create negative samples
        self.train_neg_edges = self.sample_negative_edges(self.train_edges)
        self.test_neg_edges = self.sample_negative_edges(self.test_edges)

    def split_edges(self, split_ratio):
        edge_list = self.edge_index.t().tolist()
        random.shuffle(edge_list)

        split = int(len(edge_list) * split_ratio)
        train_edges = edge_list[:split]
        test_edges = edge_list[split:]

        return torch.tensor(train_edges).t(), torch.tensor(test_edges).t()
    
    def sample_negative_edges(self, pos_edge_index):
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=self.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )
        return neg_edge_index
    
    def get_data(self, features):
        """
        Returns PyG `Data` object with features and train edges.
        """
        data = Data(x=features, edge_index=self.edge_index)
        return data
    
    
