# File: data_preprocessing.py
from colorama import Fore, Style

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