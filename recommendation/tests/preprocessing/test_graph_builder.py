import torch
import pandas as pd
from src.preprocessing.graph_builder import build_bipartite_graph, save_edge_index

def test_bipartite_graph_creation():
    # Sample DataFrame with user-item interactions
    data = {
        'reviewerID': ['U1', 'U2', 'U3', 'U1'],
        'asin': ['P1', 'P2', 'P1', 'P3'],
    }
    df = pd.DataFrame(data)

    # Build the bipartite graph
    edge_index = build_bipartite_graph(df, 'reviewerID', 'asin')

    # Test that edge_index is a torch tensor
    assert isinstance(edge_index, torch.Tensor)

    # Test the dimensions of edge_index: Should be [2, N], where N is the number of edges (interactions)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == len(df)

    # Test the values in edge_index (user-item interactions)
    assert edge_index[0].tolist() == [0, 1, 2, 0]  # User IDs should be mapped to [0, 1, 2, 0]
    assert edge_index[1].tolist() == [0, 1, 0, 2]  # Item IDs should be mapped to [0, 1, 0, 2]

def test_save_edge_index():
    # Sample edge_index for testing save
    edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    
    # Save the edge_index tensor to a temporary file
    file_path = 'data/test_edge_index.pt'
    save_edge_index(edge_index, file_path)

    # Load the tensor back from the file and verify
    loaded_edge_index = torch.load(file_path)

    # Test the saved edge_index is the same as the original
    assert torch.equal(edge_index, loaded_edge_index)

    # Clean up
    import os
    os.remove(file_path)
