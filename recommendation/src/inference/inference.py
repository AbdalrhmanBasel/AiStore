import torch

def inference(model, data, edge_index):
    """
    Perform inference on the graph to predict the links (edges).
    Arguments:
    - model: The trained GraphSAGE model.
    - data: The graph data object that contains node features and edge information.
    - edge_index: The edge index for which we need to predict the existence of links.

    Returns:
    - predictions: Predicted probabilities for the edges.
    """
    model.eval()
    with torch.no_grad():
        # Get node embeddings
        z = model(data.x, data.edge_index)
        
    # Dot product of node embeddings for link prediction
    src, dst = edge_index
    edge_embeddings = (z[src] * z[dst]).sum(dim=1)  # Dot product
    
    # Calculate probabilities using sigmoid function
    probs = torch.sigmoid(edge_embeddings)
    
    return probs

