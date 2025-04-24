# File: data_preprocessing.py
import torch
import random
from torch_geometric.data import Data
from src._0_data_preprocessing.graph_construction.report_graph import report_graph_details
import torch_geometric
from colorama import Fore, Style
import os
import sys


from settings import PROCESSED_DATA_DIR

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


def add_negative_samples(edge_data, graph_data, num_samples=None):
    """
    Generate negative samples (non-existent edges) for link prediction.

    Args:
        edge_data (torch.Tensor): Positive edge indices (shape [2, num_edges]).
        graph_data (torch_geometric.data.Data): Full graph data object.
        num_samples (int, optional): Number of negative samples to generate. Defaults to the number of positive edges.

    Returns:
        torch_geometric.data.Data: Data object containing negative samples.
    """
    if num_samples is None:
        num_samples = edge_data.size(1)  # Default to the same number as positive edges

    num_nodes = graph_data.num_nodes
    all_edges = set((u.item(), v.item()) for u, v in edge_data.t())

    negative_samples = []
    while len(negative_samples) < num_samples:
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()

        # Ensure u != v and (u, v) is not a positive edge
        if u != v and (u, v) not in all_edges and (v, u) not in all_edges:
            negative_samples.append([u, v])

    # Convert negative samples to tensor
    neg_edge_index = torch.tensor(negative_samples, dtype=torch.long).t().contiguous()

    # Create a Data object for negative samples
    neg_labels = torch.zeros(neg_edge_index.shape[1], dtype=torch.float)  # Labels for negative samples
    neg_graph = torch_geometric.data.Data(
        x=graph_data.x,  # Use the same node features as the original graph
        edge_index=neg_edge_index,
        y=neg_labels,
        num_nodes=graph_data.num_nodes
    )

    return neg_graph

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

# def split_and_save_data(graph_data):
#     print(f"{Fore.CYAN}[INFO] Splitting data into training, validation, and test sets...{Style.RESET_ALL}")
#     train_edges, val_edges, test_edges = split_edges(graph_data)
    
#     train_neg = negative_sampling(graph_data, len(train_edges))
#     val_neg = negative_sampling(graph_data, len(val_edges))
#     test_neg = negative_sampling(graph_data, len(test_edges))
    
#     train_graph = create_labeled_graph(train_edges, train_neg, graph_data.num_nodes, graph_data.x)
#     val_graph = create_labeled_graph(val_edges, val_neg, graph_data.num_nodes, graph_data.x)
#     test_graph = create_labeled_graph(test_edges, test_neg, graph_data.num_nodes, graph_data.x)
    
#     torch.save(train_graph, TRAIN_DATA_PATH)
#     torch.save(val_graph, VAL_DATA_PATH)
#     torch.save(test_graph, TEST_DATA_PATH)


# def split_and_save_data(graph_data):
#     print(f"{Fore.CYAN}[INFO] Splitting data into training, validation, and test sets...{Style.RESET_ALL}")
#     train_edges, val_edges, test_edges = split_edges(graph_data)
    
#     train_neg = negative_sampling(graph_data, len(train_edges))
#     val_neg = negative_sampling(graph_data, len(val_edges))
#     test_neg = negative_sampling(graph_data, len(test_edges))
    
#     train_graph = create_labeled_graph(train_edges, train_neg, graph_data.num_nodes, graph_data.x)
#     val_graph = create_labeled_graph(val_edges, val_neg, graph_data.num_nodes, graph_data.x)
#     test_graph = create_labeled_graph(test_edges, test_neg, graph_data.num_nodes, graph_data.x)
    
#     # Save labeled graphs
#     try:
#         torch.save(train_graph, TRAIN_DATA_PATH)
#         torch.save(val_graph, VAL_DATA_PATH)
#         torch.save(test_graph, TEST_DATA_PATH)
#         print(f"{Fore.GREEN}✅ Saved split data: {TRAIN_DATA_PATH}, {VAL_DATA_PATH}, {TEST_DATA_PATH}{Style.RESET_ALL}")
#     except Exception as e:
#         print(f"{Fore.RED}[ERROR] Failed to save split data: {e}{Style.RESET_ALL}")
#         raise

# def split_and_save_data(graph_data):
#     """
#     Split graph data into training/validation/test sets and generate negative samples.
#     Combine positive and negative edges into labeled graph objects for each split.
#     """
#     print(f"\n{Fore.CYAN}[INFO] Splitting data into training, validation, and test sets...{Style.RESET_ALL}")

#     try:
#         # Split edges into training, validation, and test sets
#         train_edges, val_edges, test_edges = split_edges(graph_data)

#         # Generate negative samples for each split
#         train_neg_samples = add_negative_samples(train_edges, graph_data).edge_index
#         val_neg_samples = add_negative_samples(val_edges, graph_data).edge_index
#         test_neg_samples = add_negative_samples(test_edges, graph_data).edge_index

#         # Combine positive and negative samples for each split
#         def create_labeled_graph(edges, neg_samples, num_nodes):
#             pos_labels = torch.ones(edges.shape[1], dtype=torch.float)  # Positive labels
#             neg_labels = torch.zeros(neg_samples.shape[1], dtype=torch.float)  # Negative labels

#             # Concatenate positive and negative edges
#             all_edges = torch.cat([edges, neg_samples], dim=1)
#             all_labels = torch.cat([pos_labels, neg_labels])

#             return torch_geometric.data.Data(
#                 x=graph_data.x,
#                 edge_index=all_edges,
#                 y=all_labels,
#                 num_nodes=num_nodes
#             )

#         # Create labeled graphs for each split
#         train_graph = create_labeled_graph(train_edges, train_neg_samples, graph_data.num_nodes)
#         val_graph = create_labeled_graph(val_edges, val_neg_samples, graph_data.num_nodes)
#         test_graph = create_labeled_graph(test_edges, test_neg_samples, graph_data.num_nodes)

#         # Save labeled graphs
#         os.makedirs(os.path.join(PROCESSED_DATA_DIR, "graph"), exist_ok=True)
#         torch.save(train_graph, os.path.join(PROCESSED_DATA_DIR, "graph/train_data.pt"))
#         torch.save(val_graph, os.path.join(PROCESSED_DATA_DIR, "graph/val_data.pt"))
#         torch.save(test_graph, os.path.join(PROCESSED_DATA_DIR, "graph/test_data.pt"))

#         print(f"{Fore.GREEN}✅ Saved split data with labels.{Style.RESET_ALL}")
#     except Exception as e:
#         print(f"{Fore.RED}[ERROR] Failed to split or save data: {e}{Style.RESET_ALL}")
#         raise

#     # Report details of each split
#     print(f"\n{Fore.CYAN}[INFO] Reporting details of training graph...{Style.RESET_ALL}")
#     report_graph_details(train_graph)

#     print(f"\n{Fore.CYAN}[INFO] Reporting details of validation graph...{Style.RESET_ALL}")
#     report_graph_details(val_graph)

#     print(f"\n{Fore.CYAN}[INFO] Reporting details of test graph...{Style.RESET_ALL}")
#     report_graph_details(test_graph)

#     print(f"\n{Fore.BLUE}[INFO] Shapes of the split data:{Style.RESET_ALL}")
#     print(f"{Fore.BLUE}Training data shape: {train_graph.edge_index.shape}{Style.RESET_ALL}")
#     print(f"{Fore.BLUE}Validation data shape: {val_graph.edge_index.shape}{Style.RESET_ALL}")
#     print(f"{Fore.BLUE}Test data shape: {test_graph.edge_index.shape}{Style.RESET_ALL}")

#     print(f"{Fore.GREEN}✅ Split data into training, validation, and test sets.{Style.RESET_ALL}")
#     print(f"{Fore.GREEN}✅ Negative samples added for link prediction.{Style.RESET_ALL}")

    



# def split_and_save_data(graph_data):
#     """
#     Split graph data into training/validation/test sets and generate negative samples.
#     """
#     print(f"{Fore.CYAN}[INFO] Splitting data into training, validation, and test sets...{Style.RESET_ALL}")
#     try:
#         # Split edges into training, validation, and test sets
#         train_edges, val_edges, test_edges = split_edges(graph_data)
        
#         # Generate negative samples for each split
#         train_neg_samples = add_negative_samples(train_edges, graph_data)
#         val_neg_samples = add_negative_samples(val_edges, graph_data)
#         test_neg_samples = add_negative_samples(test_edges, graph_data)
        
#         # Combine positive and negative samples for each split
#         def create_labeled_graph(edges, neg_samples, num_nodes):
#             pos_labels = torch.ones(edges.shape[1], dtype=torch.float)  # Positive labels
#             neg_labels = torch.zeros(neg_samples.shape[1], dtype=torch.float)  # Negative labels
            
#             # Concatenate positive and negative edges
#             all_edges = torch.cat([edges, neg_samples], dim=1)
#             all_labels = torch.cat([pos_labels, neg_labels])
            
#             return torch_geometric.data.Data(
#                 x=graph_data.x,
#                 edge_index=all_edges,
#                 y=all_labels,
#                 num_nodes=num_nodes
#             )
        
#         # Create labeled graphs for each split
#         train_graph = create_labeled_graph(train_edges, train_neg_samples, graph_data.num_nodes)
#         val_graph = create_labeled_graph(val_edges, val_neg_samples, graph_data.num_nodes)
#         test_graph = create_labeled_graph(test_edges, test_neg_samples, graph_data.num_nodes)
        
#         # Save positive edges and negative samples
#         os.makedirs(os.path.join(PROCESSED_DATA_DIR, "graph/positives"), exist_ok=True)
#         os.makedirs(os.path.join(PROCESSED_DATA_DIR, "graph/negatives"), exist_ok=True)
        
#         torch.save(train_graph, os.path.join(PROCESSED_DATA_DIR, "graph/positives/train_data.pt"))
#         torch.save(val_graph, os.path.join(PROCESSED_DATA_DIR, "graph/positives/val_data.pt"))
#         torch.save(test_graph, os.path.join(PROCESSED_DATA_DIR, "graph/positives/test_data.pt"))
        
#         torch.save(train_neg_samples, os.path.join(PROCESSED_DATA_DIR, "graph/negatives/train_neg_samples.pt"))
#         torch.save(val_neg_samples, os.path.join(PROCESSED_DATA_DIR, "graph/negatives/val_neg_samples.pt"))
#         torch.save(test_neg_samples, os.path.join(PROCESSED_DATA_DIR, "graph/negatives/test_neg_samples.pt"))
        
#         print(f"{Fore.GREEN}✅ Saved split data with labels.{Style.RESET_ALL}")
#     except Exception as e:
#         print(f"{Fore.RED}[ERROR] Failed to split or save data: {e}{Style.RESET_ALL}")
#         raise


def split_and_save_data(graph_data):
    """
    Split graph data into training/validation/test sets and generate negative samples.
    Combine positive and negative edges into labeled graph objects for each split,
    then save both the labeled graphs and raw negative samples under separate directories.
    """
    print(f"\n{Fore.CYAN}[INFO] Splitting data into training, validation, and test sets...{Style.RESET_ALL}")

    try:
        # Split edges into training, validation, and test sets
        train_edges, val_edges, test_edges = split_edges(graph_data)

        # Generate negative samples for each split (Data objects)
        train_neg = add_negative_samples(train_edges, graph_data)
        val_neg = add_negative_samples(val_edges, graph_data)
        test_neg = add_negative_samples(test_edges, graph_data)

        # Extract edge_index tensors for labeling
        train_neg_edges = train_neg.edge_index
        val_neg_edges = val_neg.edge_index
        test_neg_edges = test_neg.edge_index

        # Helper to build labeled Data
        def create_labeled_graph(pos_edges, neg_edges, num_nodes):
            pos_labels = torch.ones(pos_edges.shape[1], dtype=torch.float)
            neg_labels = torch.zeros(neg_edges.shape[1], dtype=torch.float)
            all_edges = torch.cat([pos_edges, neg_edges], dim=1)
            all_labels = torch.cat([pos_labels, neg_labels])
            return torch_geometric.data.Data(
                x=graph_data.x,
                edge_index=all_edges,
                y=all_labels,
                num_nodes=num_nodes
            )

        # Create labeled graph objects
        train_graph = create_labeled_graph(train_edges, train_neg_edges, graph_data.num_nodes)
        val_graph   = create_labeled_graph(val_edges,   val_neg_edges,   graph_data.num_nodes)
        test_graph  = create_labeled_graph(test_edges,  test_neg_edges,  graph_data.num_nodes)

        # Prepare directories
        pos_dir = os.path.join(PROCESSED_DATA_DIR, "graph/positives")
        neg_dir = os.path.join(PROCESSED_DATA_DIR, "graph/negatives")
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        # Save labeled graphs
        torch.save(train_graph, os.path.join(pos_dir, "train_data.pt"))
        torch.save(val_graph,   os.path.join(pos_dir, "val_data.pt"))
        torch.save(test_graph,  os.path.join(pos_dir, "test_data.pt"))

        # Save raw negative samples
        torch.save(train_neg, os.path.join(neg_dir, "train_neg_samples.pt"))
        torch.save(val_neg,   os.path.join(neg_dir, "val_neg_samples.pt"))
        torch.save(test_neg,  os.path.join(neg_dir, "test_neg_samples.pt"))

        print(f"{Fore.GREEN}✅ Saved split data and negative samples under 'graph/positives' and 'graph/negatives'.{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to split or save data: {e}{Style.RESET_ALL}")
        raise

    # Reporting for each split
    for name, g in [('Training', train_graph), ('Validation', val_graph), ('Test', test_graph)]:
        print(f"\n{Fore.CYAN}[INFO] Reporting details of {name.lower()} graph...{Style.RESET_ALL}")
        report_graph_details(g)

    print(f"\n{Fore.BLUE}[INFO] Shapes of the split data:{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Training edges: {train_graph.edge_index.shape}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Validation edges: {val_graph.edge_index.shape}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Test edges: {test_graph.edge_index.shape}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✅ Completed splitting and saving data with labels and negative samples.{Style.RESET_ALL}")
