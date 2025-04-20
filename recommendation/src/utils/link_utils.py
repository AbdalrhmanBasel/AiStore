import torch

def decode_link(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Computes the dot product between node embeddings at the source and destination of the edges.
    
    Args:
        z (torch.Tensor): Node embeddings of shape (N, D).
        edge_index (torch.Tensor): Edge indices of shape (2, E), where E is the number of edges.
    
    Returns:
        torch.Tensor: A tensor of shape (E,) containing the dot product for each edge.
    """
    src, dst = edge_index
    return (z[src] * z[dst]).sum(dim=1)
