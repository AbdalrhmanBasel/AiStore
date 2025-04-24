# File: data_preprocessing.py
import torch
import numpy as np
import random
from torch_geometric.data import Data
from colorama import Fore, Style
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(PROJECT_ROOT)

from settings import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH

def split_edges(graph_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    num_edges = graph_data.num_edges
    train_size = int(num_edges * train_ratio)
    val_size = int(num_edges * val_ratio)
    # test_size = num_edges - train_size - val_size

    perm = torch.randperm(num_edges)
    train_edges = perm[:train_size]
    val_edges = perm[train_size:train_size + val_size]
    test_edges = perm[train_size + val_size:]

    return (
        graph_data.edge_index[:, train_edges],
        graph_data.edge_index[:, val_edges],
        graph_data.edge_index[:, test_edges]
    )

def negative_sampling(graph_data, num_negative_samples):
    edge_index = graph_data.edge_index
    num_nodes = graph_data.num_nodes
    negative_edges = []

    for _ in range(num_negative_samples):
        while True:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if (
                u != v
                and not torch.any((edge_index[0] == u) & (edge_index[1] == v))
                and not torch.any((edge_index[0] == v) & (edge_index[1] == u))
            ):
                negative_edges.append([u, v])
                break

    return torch.tensor(negative_edges, dtype=torch.long).t().contiguous()

def create_labeled_graph(edges, neg_samples, num_nodes, node_features):
    all_edges = torch.cat([edges, neg_samples], dim=1)
    pos_labels = torch.ones(edges.shape[1], dtype=torch.float)
    neg_labels = torch.zeros(neg_samples.shape[1], dtype=torch.float)
    all_labels = torch.cat([pos_labels, neg_labels])
    
    return Data(
        x=node_features,
        edge_index=all_edges,
        y=all_labels,
        num_nodes=num_nodes
    )

def split_and_save_data(graph_data):
    print(f"{Fore.CYAN}[INFO] Splitting data into training, validation, and test sets...{Style.RESET_ALL}")
    train_edges, val_edges, test_edges = split_edges(graph_data)
    
    train_neg = negative_sampling(graph_data, len(train_edges))
    val_neg = negative_sampling(graph_data, len(val_edges))
    test_neg = negative_sampling(graph_data, len(test_edges))
    
    train_graph = create_labeled_graph(train_edges, train_neg, graph_data.num_nodes, graph_data.x)
    val_graph = create_labeled_graph(val_edges, val_neg, graph_data.num_nodes, graph_data.x)
    test_graph = create_labeled_graph(test_edges, test_neg, graph_data.num_nodes, graph_data.x)
    
    torch.save(train_graph, TRAIN_DATA_PATH)
    torch.save(val_graph, VAL_DATA_PATH)
    torch.save(test_graph, TEST_DATA_PATH)