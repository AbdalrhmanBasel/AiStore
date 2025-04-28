from torch_geometric.loader import NeighborLoader

def create_data_loaders(graph, batch_size=512, num_neighbors=[10,5]):
    train_loader = NeighborLoader(
        graph,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=graph.train_mask,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        graph,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=graph.val_mask
    )
    
    return train_loader, val_loader