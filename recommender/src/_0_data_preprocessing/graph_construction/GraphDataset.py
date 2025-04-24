import torch
from torch_geometric.data import Data


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