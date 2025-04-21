from torch_geometric.data import Data
import torch 

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        """
        Initialize the dataset by loading the graph data from the specified path.
        
        Args:
            path (str): Path to the .pt file containing the graph data.
        """
        self.data = torch.load(path)  # Load the .pt file

    def __len__(self) -> int:
        return self.data.edge_index.shape[1]  # Number of edges

    def __getitem__(self, idx: int) -> Data:
        # Node features
        x = self.data.x
        
        # Single edge index
        edge_index = self.data.edge_index[:, idx].unsqueeze(1)  # Shape: [2, 1]
        
        # Label (if available)
        y = self.data.y[idx] if hasattr(self.data, "y") and self.data.y is not None else None
        
        # Return as a PyTorch Geometric Data object
        return Data(x=x, edge_index=edge_index, y=y)