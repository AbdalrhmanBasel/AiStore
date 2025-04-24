import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader  
from typing import Dict

from src._1_model_selection.GraphSAGEModelV0 import GraphSAGEModelV0
import torch_geometric
from settings import PROCESSED_DATA_DIR, CHECKPOINT_DIR, MODEL_NAME, ENABLE_CUDA, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, NUM_LAYERS, DROPOUT_RATE

def link_prediction_loss(pos_edge_index, neg_edge_index, z):
    # pos_edge_index: positive edge indices (real edges in the graph)
    # neg_edge_index: negative edge indices (non-existing edges)
    
    # Get node embeddings for positive and negative edges
    pos_edge_emb = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_edge_emb = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    
    # Positive edge score (the higher, the more likely it is a real edge)
    pos_loss = F.logsigmoid(pos_edge_emb).mean()
    
    # Negative edge score (the lower, the more likely it is a fake edge)
    neg_loss = F.logsigmoid(-neg_edge_emb).mean()
    
    return -(pos_loss + neg_loss)


def precision_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    Compute precision@k for the given scores and labels.
    """
    _, indices = torch.topk(scores, k)
    top_k_labels = labels[indices]
    precision = top_k_labels.sum().item() / k
    return precision


def recall_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    Compute recall@k for the given scores and labels.
    """
    _, indices = torch.topk(scores, k)
    top_k_labels = labels[indices]
    recall = top_k_labels.sum().item() / labels.sum().item()
    return recall


def ndcg_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) for the given scores and labels.
    """
    _, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]
    dcg = (sorted_labels / torch.log2(torch.arange(len(sorted_labels), dtype=torch.float) + 2)).sum().item()
    ideal_labels, _ = torch.sort(labels, descending=True)
    idcg = (ideal_labels / torch.log2(torch.arange(len(ideal_labels), dtype=torch.float) + 2)).sum().item()
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_training_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,  # Add loss_fn as an argument
    device: str,
) -> Dict[str, float]:
    """
    Evaluate the model on the validation dataset.
    This function calculates ranking metrics like precision@k, recall@k, and NDCG.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (DataLoader): DataLoader for the validation set.
        loss_fn (torch.nn.Module): The loss function to use for evaluation.
        device (str): The device to run the evaluation on (e.g., "cuda" or "cpu").

    Returns:
        Dict[str, float]: A dictionary containing the loss and ranking metrics.
    """
    model.eval()
    all_scores, all_labels = [], []

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data.x, data.edge_index)

            # Compute edge scores (dot product between connected nodes)
            src_embeddings = output[data.edge_index[0]]
            dst_embeddings = output[data.edge_index[1]]
            edge_scores = (src_embeddings * dst_embeddings).sum(dim=1)

            # Collect scores and labels
            all_scores.append(edge_scores.cpu())
            all_labels.append(data.y.cpu())

    # Concatenate scores and labels
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    # Compute ranking metrics
    precision = precision_at_k(all_scores, all_labels, k=10)
    recall = recall_at_k(all_scores, all_labels, k=10)
    ndcg = ndcg_score(all_scores, all_labels)

    # Compute loss using the provided loss_fn
    loss = loss_fn(all_scores, all_labels.float()).item()

    return {
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg,
    }


def evaluate():
    """
    Function to evaluate the trained model.
    Includes steps like model evaluation on test data and performance metrics calculation.
    """
    print("Evaluating the model...")

    # Step 1: Load test data
    try:
        test_data_path = os.path.join(PROCESSED_DATA_DIR, "graph/positives/test_data.pt")
        test_neg_samples_path = os.path.join(PROCESSED_DATA_DIR, "graph/negatives/test_neg_samples.pt")

        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data file not found at {test_data_path}")
        if not os.path.exists(test_neg_samples_path):
            raise FileNotFoundError(f"Negative samples file not found at {test_neg_samples_path}")

        test_data = torch.load(test_data_path, weights_only=False)
        test_neg_samples = torch.load(test_neg_samples_path, weights_only=False)

        # Debugging: Print the type and structure of loaded data
        print(f"Type of test_data: {type(test_data)}")
        print(f"Type of test_neg_samples: {type(test_neg_samples)}")

        # Ensure the loaded data is of the correct type
        if not isinstance(test_data, torch_geometric.data.Data):
            raise TypeError("test_data must be an instance of torch_geometric.data.Data")
        if not isinstance(test_neg_samples, torch_geometric.data.Data):
            # Attempt to convert test_neg_samples to Data if it's a dictionary
            if isinstance(test_neg_samples, dict):
                test_neg_samples = torch_geometric.data.Data(
                    x=test_neg_samples.get("x", None),
                    edge_index=test_neg_samples.get("edge_index", None),
                    y=test_neg_samples.get("y", None),
                )
            else:
                raise TypeError("test_neg_samples must be an instance of torch_geometric.data.Data")

    except Exception as e:
        print(f"Error loading test data: {e}")
        raise

    # Step 2: Load the trained model
    model_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    try:
        # Load the checkpoint (contains state_dict and metadata)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Initialize the model architecture
        input_dim = test_data.num_node_features  # Get input dimensions from test data
        model = GraphSAGEModelV0(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT_RATE,
        )

        # Load the model weights from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])

    except Exception as e:
        print(f"Error loading the model: {e}")
        raise

    # Move the model to the appropriate device
    device = "cuda" if ENABLE_CUDA and torch.cuda.is_available() else "cpu"
    model.to(device)

    # Step 3: Evaluate the model
    evaluate_testing_model(
        model=model,
        test_data=test_data,
        test_neg_samples=test_neg_samples,
        device=device,
    )

def evaluate_testing_model(
    model: torch.nn.Module,
    test_data: torch.Tensor,
    test_neg_samples: torch.Tensor,
    device: str,
) -> None:
    """
    Evaluate the model on the test dataset.
    This function calculates ranking metrics like precision@k, recall@k, and NDCG.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    if not isinstance(test_data, torch_geometric.data.Data):
        raise TypeError("test_data must be an instance of torch_geometric.data.Data")
    if not isinstance(test_neg_samples, torch_geometric.data.Data):
        raise TypeError("test_neg_samples must be an instance of torch_geometric.data.Data")

    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        combined_data = torch.cat([test_data, test_neg_samples], dim=0).to(device)
        labels = torch.cat([torch.ones(len(test_data)), torch.zeros(len(test_neg_samples))]).to(device)

        output = model(combined_data.x, combined_data.edge_index)

        src_embeddings = F.normalize(output[combined_data.edge_index[0]], p=2, dim=1)
        dst_embeddings = F.normalize(output[combined_data.edge_index[1]], p=2, dim=1)
        edge_scores = (src_embeddings * dst_embeddings).sum(dim=1)

        all_scores.append(edge_scores.cpu())
        all_labels.append(labels.cpu())

    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    precision = precision_at_k(all_scores, all_labels, k=10)
    recall = recall_at_k(all_scores, all_labels, k=10)
    ndcg = ndcg_score(all_scores, all_labels)

    print("Evaluation Results:")
    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10: {recall:.4f}")
    print(f"NDCG: {ndcg:.4f}")