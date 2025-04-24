import json
import torch
from torch_geometric.utils import is_undirected, contains_isolated_nodes, contains_self_loops

def show_graph_summary(graph_dataset):
    """
    Prints a graph report to the terminal.
    """

    print("=" * 50)
    print(f"📦 graph_dataset: {graph_dataset.__class__.__name__}")
    print("=" * 50)
    print(f"Graphs: {len(graph_dataset)}")

    graph = graph_dataset[0]

    print(f"\n🧠 Structure")
    print("-" * 30)
    print(f"Nodes           : {graph.num_nodes}")
    print(f"Node features   : {graph.num_node_features}")
    print(f"Edge features   : {graph.num_edge_features if 'edge_attr' in graph else 0}")
    print(f"Edges           : {graph.edge_index.shape[1]}")
    print(f"Classes         : {getattr(graph_dataset, 'num_classes', 'Unknown')}")

    print(f"\n🧪 Splits")
    print("-" * 30)
    print(f"Train nodes     : {int(graph.train_mask.sum())}")
    print(f"Val nodes       : {int(graph.val_mask.sum())}")
    print(f"Test nodes      : {int(graph.test_mask.sum())}")

    print(f"\n🔍 Properties")
    print("-" * 30)
    print(f"Undirected      : {is_undirected(graph.edge_index)}")
    print(f"Isolated nodes  : {contains_isolated_nodes(graph.edge_index, graph.num_nodes)}")
    print(f"Self-loops      : {contains_self_loops(graph.edge_index)}")
    print(f"Edge shape      : {tuple(graph.edge_index.shape)}")

    if hasattr(graph, 'y') and graph.y.dim() == 1:
        dist = torch.bincount(graph.y)
        print(f"\n📊 Classes")
        print("-" * 30)
        for i, count in enumerate(dist):
            print(f"Class {i}: {count.item()}")
    print("=" * 50)


def export_graph_summary(graph_dataset, as_format='dict', save=None):
    """
    Renders a dictionary or json based graph report.
    """

    graph = graph_dataset[0]
    r = {
        "graph_dataset": graph_dataset.__class__.__name__,
        "graphs": len(graph_dataset),
        "nodes": graph.num_nodes,
        "node_features": graph.num_node_features,
        "edge_features": graph.num_edge_features if "edge_attr" in graph else 0,
        "edges": graph.edge_index.shape[1],
        "classes": getattr(graph_dataset, 'num_classes', 'Unknown'),
        "train_nodes": int(graph.train_mask.sum()),
        "val_nodes": int(graph.val_mask.sum()),
        "test_nodes": int(graph.test_mask.sum()),
        "undirected": is_undirected(graph.edge_index),
        "isolated_nodes": contains_isolated_nodes(graph.edge_index, graph.num_nodes),
        "self_loops": contains_self_loops(graph.edge_index),
        "edge_shape": tuple(graph.edge_index.shape),
    }

    if hasattr(graph, 'y') and graph.y.dim() == 1:
        dist = torch.bincount(graph.y)
        r["class_distribution"] = {f"class_{i}": count.item() for i, count in enumerate(dist)}
    else:
        r["class_distribution"] = "Not available"

    if as_format == 'json':
        result = json.dumps(r, indent=4)
        if save:
            with open(save, 'w') as f:
                f.write(result)
        return result
    return r
