import torch
from torch.utils.data import Dataset
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import LabelEncoder
from logger import get_module_logger

import os 
import sys 
logger = get_module_logger("dataset")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)

class GraphDataset(Dataset):
    def __init__(self, reviews_df, metadata_df, max_nodes=None):
        """
        Initialize GraphDataset class for graph-based recommendation system.

        :param reviews_df: DataFrame containing the reviews data (user_id, parent_asin, rating, timestamp)
        :param metadata_df: DataFrame containing metadata (asin, category, title, etc.)
        :param max_nodes: The maximum number of nodes to consider in the graph (optional)
        """
        # Store the input data
        self.reviews_df = reviews_df
        self.metadata_df = metadata_df

        # Encode user_id and parent_asin into integers (node IDs)
        self.user_encoder = LabelEncoder()
        self.asin_encoder = LabelEncoder()
        
        self.reviews_df['user_id_encoded'] = self.user_encoder.fit_transform(self.reviews_df['user_id'])
        self.reviews_df['parent_asin_encoded'] = self.asin_encoder.fit_transform(self.reviews_df['parent_asin'])
        
        # Create Graph
        self.graph = nx.Graph()

        # Add nodes for users and products (ASIN)
        self.graph.add_nodes_from(self.reviews_df['user_id_encoded'])
        self.graph.add_nodes_from(self.reviews_df['parent_asin_encoded'])

        # Add edges based on reviews
        for _, row in self.reviews_df.iterrows():
            user_node = row['user_id_encoded']
            asin_node = row['parent_asin_encoded']
            self.graph.add_edge(user_node, asin_node, rating=row['rating'])

        # Create metadata features for products (e.g., categories, prices)
        self.product_metadata = metadata_df.set_index('parent_asin').to_dict(orient='index')

        # Limit the graph to a maximum number of nodes (optional)
        if max_nodes:
            nodes_to_keep = list(self.graph.nodes)[:max_nodes]
            self.graph = self.graph.subgraph(nodes_to_keep)

    def __len__(self):
        return len(self.reviews_df)

    def __getitem__(self, idx):
        """
        Returns the data for a single sample.
        
        :param idx: Index of the sample to return
        :return: A dictionary with node features and edges for the GNN model
        """
        row = self.reviews_df.iloc[idx]
        user_node = row['user_id_encoded']
        asin_node = row['parent_asin_encoded']
        rating = row['rating']
        
        # Fetch metadata for the product (ASIN)
        asin_metadata = self.product_metadata.get(row['parent_asin'], {})

        # Example of possible metadata features to include
        metadata_features = [
            asin_metadata.get('main_category', 'unknown'),
            asin_metadata.get('price', 0.0),
            len(asin_metadata.get('features', [])),
            len(asin_metadata.get('categories', []))
        ]

        # Convert metadata to numerical features (e.g., using LabelEncoder or one-hot encoding)
        metadata_features = np.array(metadata_features, dtype=np.float32)

        # Create a dictionary of the node features
        user_features = torch.tensor(metadata_features, dtype=torch.float32)  # Example, you can replace it with actual user features

        edge_index = torch.tensor([[user_node, asin_node]], dtype=torch.long).T  # Edge between user and product
        edge_attr = torch.tensor([rating], dtype=torch.float32)  # Rating as edge feature
        
        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'user_features': user_features,
            'asin_node': asin_node
        }

    def get_node_features(self):
        """
        Generate features for all nodes.
        """
        node_features = {}
        for node in self.graph.nodes:
            # Extract features based on whether it's a user or an item (ASIN)
            if node in self.user_encoder.classes_:
                node_features[node] = torch.tensor([1.0], dtype=torch.float32)  # Example, add user-specific features
            else:
                asin_metadata = self.product_metadata.get(self.asin_encoder.inverse_transform([node])[0], {})
                features = [
                    asin_metadata.get('price', 0.0),
                    len(asin_metadata.get('features', [])),
                    len(asin_metadata.get('categories', [])),
                ]
                node_features[node] = torch.tensor(features, dtype=torch.float32)
        
        return node_features
