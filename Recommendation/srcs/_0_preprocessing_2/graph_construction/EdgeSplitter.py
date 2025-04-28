from logger import get_module_logger
import os
import torch
import random
import numpy as np
from torch_geometric.data import Data
import torch_geometric
from typing import Tuple
from collections import defaultdict
from settings import PROCESSED_DATA_DIR

logger = get_module_logger("split_graph")

class EdgeSplitter:
    """Advanced edge splitting with stratification"""
    
    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.edge_set = set((u.item(), v.item()) for u, v in edge_index.t())
        
    def stratified_split(self, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """Split edges with degree stratification"""
        degrees = defaultdict(int)
        for u, v in self.edge_index.t():
            degrees[u.item()] += 1
            degrees[v.item()] += 1
            
        # Bin edges by degree
        bins = {
            'low': [],
            'medium': [],
            'high': []
        }
        
        for idx, (u, v) in enumerate(self.edge_index.t()):
            avg_degree = (degrees[u.item()] + degrees[v.item()]) / 2
            if avg_degree < 5:
                bins['low'].append(idx)
            elif avg_degree < 20:
                bins['medium'].append(idx)
            else:
                bins['high'].append(idx)
                
        # Sample from each bin proportionally
        train_idx, val_idx, test_idx = [], [], []
        for bin_name, indices in bins.items():
            np.random.shuffle(indices)
            n = len(indices)
            train_idx.extend(indices[:int(n*ratios[0])])
            val_idx.extend(indices[int(n*ratios[0]):int(n*(ratios[0]+ratios[1]))])
            test_idx.extend(indices[int(n*(ratios[0]+ratios[1])):])
            
        return (
            self.edge_index[:, train_idx],
            self.edge_index[:, val_idx],
            self.edge_index[:, test_idx]
        )

def split_and_save_data(graph_data: Data):
    """Enhanced splitting with negative sampling"""
    logger.info("Starting data splitting")
    
    try:
        # 1. Initialize splitter
        splitter = EdgeSplitter(graph_data.edge_index, graph_data.num_nodes)
        
        # 2. Stratified splitting
        train_edges, val_edges, test_edges = splitter.stratified_split()
        logger.info(f"Split sizes - Train: {train_edges.shape[1]}, Val: {val_edges.shape[1]}, Test: {test_edges.shape[1]}")
        
        # 3. Negative sampling (hard mining)
        def sample_negatives(pos_edges, num_samples):
            degrees = torch_geometric.utils.degree(graph_data.edge_index[0])
            prob = degrees / degrees.sum()
            
            neg_samples = []
            while len(neg_samples) < num_samples:
                u = torch.multinomial(prob, 1).item()
                v = random.randint(0, graph_data.num_nodes - 1)
                if (u, v) not in splitter.edge_set and u != v:
                    neg_samples.append([u, v])
            return torch.tensor(neg_samples, dtype=torch.long).t()
        
        # 4. Create and save splits
        os.makedirs(f"{PROCESSED_DATA_DIR}/graph", exist_ok=True)
        
        for name, pos_edges in [('train', train_edges), ('val', val_edges), ('test', test_edges)]:
            neg_edges = sample_negatives(pos_edges, pos_edges.shape[1])
            
            # Create labeled graph
            all_edges = torch.cat([pos_edges, neg_edges], dim=1)
            labels = torch.cat([
                torch.ones(pos_edges.shape[1]),
                torch.zeros(neg_edges.shape[1])
            ])
            
            split_graph = Data(
                x=graph_data.x,edge_index=all_edges,
                y=labels,num_nodes=graph_data.num_nodes
            )
            
            torch.save(split_graph, f"{PROCESSED_DATA_DIR}/graph/{name}_data.pt")
            logger.info("{name}_data.pt has been saved successfully")

            torch.save(neg_edges, f"{PROCESSED_DATA_DIR}/graph/{name}_neg_samples.pt")
            logger.info("{name}_neg_samples.pt has been saved successfully")
            
        logger.info("Data splitting completed successfully")
        
    except Exception as e:
        logger.error("Data splitting failed", exc_info=True)
        raise