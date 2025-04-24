
import torch
import numpy as np
import random
from torch_geometric.data import Data
import torch_geometric

from colorama import Fore, Style, init

# Initialize colorama for cross-platform compatibility
init()

def split_edges(graph_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the edges of a graph into training, validation, and test sets.

    Args:
        graph_data (torch_geometric.data.Data): Input graph data object.
        train_ratio (float): Proportion of edges for training (default: 0.8).
        val_ratio (float): Proportion of edges for validation (default: 0.1).
        test_ratio (float): Proportion of edges for testing (default: 0.1).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Edge indices for training, validation, and test sets.
    """
    num_edges = graph_data.num_edges
    train_size = int(num_edges * train_ratio)
    val_size = int(num_edges * val_ratio)
    test_size = num_edges - train_size - val_size

    # Randomly shuffle edges for splitting
    perm = torch.randperm(num_edges)
    train_edges = perm[:train_size]
    val_edges = perm[train_size:train_size + val_size]
    test_edges = perm[train_size + val_size:]

    # Create new edge index tensors for train, val, and test
    train_data = graph_data.edge_index[:, train_edges]
    val_data = graph_data.edge_index[:, val_edges]
    test_data = graph_data.edge_index[:, test_edges]

    return train_data, val_data, test_data


def negative_sampling(graph_data, num_negative_samples):
    """
    Generates negative edges (non-existent edges) for link prediction.

    Args:
        graph_data (torch_geometric.data.Data): Input graph data object.
        num_negative_samples (int): Number of negative samples to generate.

    Returns:
        torch.Tensor: Tensor of shape [2, num_negative_samples] containing negative edges.
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
            if (
                u != v
                and not torch.any((edge_index[0] == u) & (edge_index[1] == v))
                and not torch.any((edge_index[0] == v) & (edge_index[1] == u))
            ):
                negative_edges.append([u, v])
                break

    # Convert negative edges to tensor
    negative_edges = torch.tensor(negative_edges, dtype=torch.long).t().contiguous()

    return negative_edges


def create_data_for_link_prediction(graph_data, num_negative_samples=10000):
    """
    Prepares data for link prediction task by adding negative samples.

    Args:
        graph_data (torch_geometric.data.Data): Input graph data object.
        num_negative_samples (int): Number of negative samples to generate (default: 10000).

    Returns:
        torch_geometric.data.Data: Modified graph data object with positive and negative edges labeled.
    """
    # Generate negative samples
    negative_edges = negative_sampling(graph_data, num_negative_samples)

    # Combine positive and negative edges
    positive_edges = graph_data.edge_index
    all_edges = torch.cat([positive_edges, negative_edges], dim=1)

    # Labels: 1 for positive edges (real edges), 0 for negative edges
    edge_labels = torch.cat(
        [
            torch.ones(positive_edges.size(1), dtype=torch.float),
            torch.zeros(negative_edges.size(1), dtype=torch.float),
        ],
        dim=0,
    )

    # Return modified graph data with additional info for link prediction
    data = Data(x=graph_data.x, edge_index=all_edges, edge_attr=edge_labels)

    return data


# def add_negative_samples(edge_data, graph_data, num_samples=None):
#     """
#     Generate negative samples (non-existent edges) for link prediction.

#     Args:
#         edge_data (torch.Tensor): Positive edge indices (shape [2, num_edges]).
#         graph_data (torch_geometric.data.Data): Full graph data object.
#         num_samples (int, optional): Number of negative samples to generate. Defaults to the number of positive edges.

#     Returns:
#         torch.Tensor: Tensor of negative edge indices (shape [2, num_samples]).
#     """
#     if num_samples is None:
#         num_samples = edge_data.size(1)  # Default to the same number as positive edges

#     num_nodes = graph_data.num_nodes
#     all_edges = set((u.item(), v.item()) for u, v in edge_data.t())

#     negative_samples = []
#     while len(negative_samples) < num_samples:
#         u = torch.randint(0, num_nodes, (1,)).item()
#         v = torch.randint(0, num_nodes, (1,)).item()

#         # Ensure u != v and (u, v) is not a positive edge
#         if u != v and (u, v) not in all_edges and (v, u) not in all_edges:
#             negative_samples.append([u, v])

#     return torch.tensor(negative_samples, dtype=torch.long).t().contiguous()


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



# def add_negative_samples(edge_data, graph_data, num_samples=None):
#     """
#     Generate negative samples (non-existent edges) for link prediction.

#     Args:
#         edge_data (torch.Tensor): Positive edge indices (shape [2, num_edges]).
#         graph_data (torch_geometric.data.Data): Full graph data object.
#         num_samples (int, optional): Number of negative samples to generate. Defaults to the number of positive edges.

#     Returns:
#         torch_geometric.data.Data: Data object containing negative samples.
#     """
#     if num_samples is None:
#         num_samples = edge_data.size(1)  # Default to the same number as positive edges

#     num_nodes = graph_data.num_nodes
#     all_edges = set((u.item(), v.item()) for u, v in edge_data.t())

#     negative_samples = []
#     while len(negative_samples) < num_samples:
#         u = torch.randint(0, num_nodes, (1,)).item()
#         v = torch.randint(0, num_nodes, (1,)).item()

#         # Ensure u != v and (u, v) is not a positive edge
#         if u != v and (u, v) not in all_edges and (v, u) not in all_edges:
#             negative_samples.append([u, v])

#     # Convert negative samples to tensor
#     neg_edge_index = torch.tensor(negative_samples, dtype=torch.long).t().contiguous()

#     # Create a Data object for negative samples
#     neg_labels = torch.zeros(neg_edge_index.shape[1], dtype=torch.float)  # Labels for negative samples
#     neg_graph = torch_geometric.data.Data(
#         x=graph_data.x,  # Use the same node features as the original graph
#         edge_index=neg_edge_index,
#         y=neg_labels,
#         num_nodes=graph_data.num_nodes
#     )

#     return neg_graph







def report_graph_details(graph_data):
    """
    Reports detailed information about a graph, including all attributes and features.

    Args:
        graph_data (torch_geometric.data.Data): Input graph data object.
    """
    print(f"{Fore.CYAN}[INFO] Reporting graph details...{Style.RESET_ALL}")

    # General graph properties
    print(f"{Fore.BLUE}Nodes: {graph_data.num_nodes}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Edges: {graph_data.num_edges}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Features per node: {graph_data.num_node_features if graph_data.num_node_features else 0}{Style.RESET_ALL}")

    # Node features
    if hasattr(graph_data, "x") and graph_data.x is not None:
        print(f"{Fore.GREEN}Node features shape: {graph_data.x.shape}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Node features preview:{Style.RESET_ALL}")
        print(graph_data.x[:5])  # Show the first 5 rows of node features
    else:
        print(f"{Fore.YELLOW}No node features found.{Style.RESET_ALL}")

    # Edge index
    if hasattr(graph_data, "edge_index") and graph_data.edge_index is not None:
        print(f"{Fore.GREEN}Edge index shape: {graph_data.edge_index.shape}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Edge index preview:{Style.RESET_ALL}")
        print(graph_data.edge_index[:, :5])  # Show the first 5 edges
    else:
        print(f"{Fore.YELLOW}No edge index found.{Style.RESET_ALL}")

    # Edge attributes
    if hasattr(graph_data, "edge_attr") and graph_data.edge_attr is not None:
        print(f"{Fore.GREEN}Edge attributes shape: {graph_data.edge_attr.shape}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Edge attributes preview:{Style.RESET_ALL}")
        print(graph_data.edge_attr[:5])  # Show the first 5 edge attributes
    else:
        print(f"{Fore.YELLOW}No edge attributes found.{Style.RESET_ALL}")

    # Graph-level labels
    if hasattr(graph_data, "y") and graph_data.y is not None:
        print(f"{Fore.GREEN}Graph-level labels: {graph_data.y}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}No graph-level labels found.{Style.RESET_ALL}")

    # Masks (e.g., train_mask, val_mask, test_mask)
    masks = ["train_mask", "val_mask", "test_mask"]
    for mask in masks:
        if hasattr(graph_data, mask) and getattr(graph_data, mask) is not None:
            mask_tensor = getattr(graph_data, mask)
            print(f"{Fore.GREEN}{mask.capitalize()} shape: {mask_tensor.shape}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{mask.capitalize()} preview: {mask_tensor[:5]}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No {mask} found.{Style.RESET_ALL}")

    # Additional metadata (if any)
    print(f"{Fore.CYAN}[INFO] Additional metadata in graph:{Style.RESET_ALL}")
    for key, value in graph_data:
        if key not in ["x", "edge_index", "edge_attr", "y"] + masks:
            print(f"{Fore.GREEN}{key}: {value}{Style.RESET_ALL}")

    print(f"{Fore.CYAN}[INFO] Graph details reported successfully.{Style.RESET_ALL}")

# ==== Example Usage ====

if __name__ == "__main__":
    """
    Example usage of the edge splitting and link prediction data preparation pipeline.

    Creates a synthetic graph, splits its edges, generates negative samples, and prepares data for link prediction.
    """
    # Create a synthetic graph
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float)  # Node features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)  # Edge indices
    graph_data = Data(x=x, edge_index=edge_index)

    # Step 1: Split edges
    train_edges, val_edges, test_edges = split_edges(graph_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print("Train edges:", train_edges)
    print("Validation edges:", val_edges)
    print("Test edges:", test_edges)

    # Step 2: Generate negative samples and prepare data for link prediction
    link_prediction_data = create_data_for_link_prediction(graph_data, num_negative_samples=5)
    print("Edge index with negative samples:", link_prediction_data.edge_index)
    print("Edge labels:", link_prediction_data.edge_attr)