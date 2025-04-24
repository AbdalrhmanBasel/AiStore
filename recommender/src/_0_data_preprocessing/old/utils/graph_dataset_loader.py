import torch
from torch_geometric.data import Data

# class GraphDataset(torch.utils.data.Dataset):
#     def __init__(self, path: str):
#         """
#         Initialize the dataset by loading the graph data from the specified path.
#         Args:
#             path (str): Path to the .pt file containing the graph data.
#         """
#         # Load the .pt file with weights_only=False (if trusted source)
#         self.data = torch.load(path, weights_only=False)

#         # Expose important attributes from the graph data
#         self.num_node_features = self.data.num_node_features if hasattr(self.data, 'num_node_features') else None
#         self.num_nodes = self.data.num_nodes if hasattr(self.data, 'num_nodes') else None
#         self.edge_index = self.data.edge_index if hasattr(self.data, 'edge_index') else None

#     def __len__(self) -> int:
#         return self.data.edge_index.shape[1]  # Number of edges

#     def __getitem__(self, idx: int) -> Data:
#         # Node features
#         x = self.data.x
#         # Single edge index
#         edge_index = self.data.edge_index[:, idx].unsqueeze(1)  # Shape: [2, 1]
#         # Label (if available)
#         y = self.data.y[idx] if hasattr(self.data, "y") and self.data.y is not None else None
#         # Return as a PyTorch Geometric Data object
#         return Data(x=x, edge_index=edge_index, y=y)
    

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.data = torch.load(path, weights_only=False)
        self.num_node_features = getattr(self.data, 'num_node_features', None)
        self.num_nodes = getattr(self.data, 'num_nodes', None)
        self.edge_index = getattr(self.data, 'edge_index', None)
        self.y = getattr(self.data, 'y', None)  # Explicitly load labels

    def __len__(self) -> int:
        return self.data.edge_index.shape[1]  # Number of edges

    def __getitem__(self, idx: int) -> Data:
        x = self.data.x
        edge_index = self.data.edge_index[:, idx].unsqueeze(1)  # Single edge
        y = self.y[idx] if self.y is not None else torch.tensor([])  # Handle missing labels
        return Data(x=x, edge_index=edge_index, y=y)