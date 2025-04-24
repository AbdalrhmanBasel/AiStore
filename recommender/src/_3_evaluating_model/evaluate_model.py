import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional
from src._2_training_model.utils.load_hyperparameters import load_best_hyperparameters
from src._1_model_selection.GraphSAGEModelV0 import GraphSAGEModelV0
from src._0_data_preprocessing.graph_construction.GraphDataset import GraphDataset
from settings import DEVICE, HYPERPARAMETERS_PATH
from torch.nn import BCEWithLogitsLoss
from torch_geometric.loader import DataLoader  
from typing import Dict
import os
import sys

from settings import (
    TRAIN_DATA_PATH,
    DEVICE,
    CHECKPOINT_DIR,
    MODEL_NAME,
    HYPERPARAMETERS_PATH,
    PROCESSED_DATA_DIR,
    
)


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../")
sys.path.append(PROJECT_ROOT)


def compute_link_prediction_loss(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Compute the loss for link prediction using positive and negative edges.
    
    Args:
        z (torch.Tensor): Node embeddings from the model.
        pos_edge_index (torch.Tensor): Positive edge indices.
        neg_edge_index (torch.Tensor): Negative edge indices.
    
    Returns:
        torch.Tensor: Combined loss for positive and negative edges.
    """
    # Compute scores for positive edges
    pos_edge_emb = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    pos_loss = -F.logsigmoid(pos_edge_emb).mean()

    # Compute scores for negative edges
    neg_edge_emb = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    neg_loss = -F.logsigmoid(-neg_edge_emb).mean()

    # Return combined loss
    return pos_loss + neg_loss


def precision_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    Compute precision@k for the given scores and labels.
    
    Args:
        scores (torch.Tensor): Predicted scores for edges.
        labels (torch.Tensor): Ground truth labels (1 for positive, 0 for negative).
        k (int): Number of top predictions to consider.
    
    Returns:
        float: Precision@k score.
    """
    _, indices = torch.topk(scores, k)
    top_k_labels = labels[indices]
    return top_k_labels.sum().item() / k


def recall_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    Compute recall@k for the given scores and labels.
    
    Args:
        scores (torch.Tensor): Predicted scores for edges.
        labels (torch.Tensor): Ground truth labels (1 for positive, 0 for negative).
        k (int): Number of top predictions to consider.
    
    Returns:
        float: Recall@k score.
    """
    _, indices = torch.topk(scores, k)
    top_k_labels = labels[indices]
    return top_k_labels.sum().item() / labels.sum().item()


def ndcg_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) for the given scores and labels.
    
    Args:
        scores (torch.Tensor): Predicted scores for edges.
        labels (torch.Tensor): Ground truth labels (1 for positive, 0 for negative).
    
    Returns:
        float: NDCG score.
    """
    _, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]

    # Compute discounted cumulative gain (DCG)
    dcg = (sorted_labels / torch.log2(torch.arange(len(sorted_labels), dtype=torch.float) + 2)).sum().item()

    # Compute ideal DCG
    ideal_labels, _ = torch.sort(labels, descending=True)
    idcg = (ideal_labels / torch.log2(torch.arange(len(ideal_labels), dtype=torch.float) + 2)).sum().item()

    # Avoid division by zero
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(
    model: torch.nn.Module,
    test_data: Data,
    test_neg_samples: Data,
    loss_fn: Optional[torch.nn.Module] = None,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate the trained model on test data and compute ranking metrics,
    filtering out any invalid edge indices to avoid CUDA asserts.
    """
    model.eval()
    model.to(device)
    test_data = test_data.to(device)
    test_neg = test_neg_samples.to(device)

    with torch.no_grad():
        # 1) Get and normalize all node embeddings
        z = model(test_data.x, test_data.edge_index)
        z = F.normalize(z, p=2, dim=1)

        # 2) Extract raw edge_index tensors
        pos_ei = test_data.edge_index
        neg_ei = test_neg.edge_index

        # 3) Build masks so we only keep edges where both endpoints < z.size(0)
        N = z.size(0)
        pos_mask = (pos_ei[0] < N) & (pos_ei[1] < N)
        neg_mask = (neg_ei[0] < N) & (neg_ei[1] < N)

        pos_ei = pos_ei[:, pos_mask]
        neg_ei = neg_ei[:, neg_mask]

        # 4) Compute dot-product scores on the filtered edges
        pos_scores = (z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)
        neg_scores = (z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)

        # 5) Move back to CPU and concatenate with labels
        scores = torch.cat([pos_scores.cpu(), neg_scores.cpu()])
        labels = torch.cat([
            torch.ones(pos_scores.size(0)),
            torch.zeros(neg_scores.size(0))
        ])

    # 6) Compute metrics
    precision = precision_at_k(scores, labels, k=10)
    recall    = recall_at_k(scores, labels, k=10)
    ndcg      = ndcg_score(scores, labels)

    # 7) Optionally compute loss over the assembled scores+labels
    loss = loss_fn(scores, labels.float()).item() if loss_fn else None

    # 8) Return all results
    return {
        "loss":      loss,
        "precision": precision,
        "recall":    recall,
        "ndcg":      ndcg,
    }


def load_test_data(test_data_path: str, test_neg_samples_path: str) -> tuple[Data, Data]:
    """
    Load test data and negative samples from disk.
    
    Args:
        test_data_path (str): Path to the test data file.
        test_neg_samples_path (str): Path to the negative samples file.
    
    Returns:
        tuple[Data, Data]: Test data and negative samples as PyTorch Geometric Data objects.
    """
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found at {test_data_path}")
    if not os.path.exists(test_neg_samples_path):
        raise FileNotFoundError(f"Negative samples file not found at {test_neg_samples_path}")

    test_data = torch.load(test_data_path, weights_only=False)
    test_neg_samples = torch.load(test_neg_samples_path, weights_only=False)

    # Ensure the loaded data is of the correct type
    if not isinstance(test_data, Data):
        raise TypeError("test_data must be an instance of torch_geometric.data.Data")
    if not isinstance(test_neg_samples, Data):
        raise TypeError("test_neg_samples must be an instance of torch_geometric.data.Data")

    return test_data, test_neg_samples





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
    Main function to evaluate a trained GNN model on test data.
    """
    try:
        # Paths to test data and negative samples
        TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "graph/positives/test_data.pt")
        TEST_NEG_SAMPLES_PATH = os.path.join(PROCESSED_DATA_DIR, "graph/negatives/test_neg_samples.pt")

        # Load test data
        print("Loading test data...")
        test_data, test_neg_samples = load_test_data(TEST_DATA_PATH, TEST_NEG_SAMPLES_PATH)

        # Load best hyperparameters
        print("Loading best hyperparameters...")
        best_hyperparameters_file = HYPERPARAMETERS_PATH
        hyperparams = load_best_hyperparameters(best_hyperparameters_file)
        print(f"Loaded Hyperparameters: {hyperparams}")

        # Extract hyperparameters
        HIDDEN_DIM = hyperparams["HIDDEN_DIM"]
        NUM_LAYERS = hyperparams["NUM_LAYERS"]
        DROPOUT_RATE = hyperparams["DROPOUT_RATE"]

        # Load training dataset to determine input dimensions
        train_dataset = GraphDataset(TRAIN_DATA_PATH)
        input_dim = train_dataset.num_node_features

        # Initialize the model
        print("Initializing the model...")
        model = GraphSAGEModelV0(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            output_dim=1,  # Assuming binary classification (output_dim=1)
            num_layers=NUM_LAYERS,
            dropout=DROPOUT_RATE,
        ).to(DEVICE)

        # Load the trained model weights
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
        print("Loading model weights...")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint['model_state_dict'])

        # Define loss function
        loss_fn = BCEWithLogitsLoss()

        # Evaluate the model
        print("Evaluating the model...")
        metrics = evaluate_model(
            model=model,
            test_data=test_data,
            test_neg_samples=test_neg_samples,
            loss_fn=loss_fn,
            device=DEVICE,
        )

        # Ensure precision and recall are added to the metrics dictionary
        precision = metrics.get('precision', None) or metrics.get('precision@10', 'N/A')
        recall = metrics.get('recall', None) or metrics.get('recall@10', 'N/A')

        # Print metrics in a horizontal table-like format
        print("\n=== Evaluation Results ===")
        print("-" * 50)
        print(f"{'Metric':<10} {'Loss':<10} {'Precision':<10} {'Recall':<10} {'NDCG':<10}")
        print("-" * 50)

        # Format each metric value properly
        loss_value = f"{metrics.get('loss', 'N/A'):<10.4f}" if isinstance(metrics.get('loss'), float) else f"{metrics.get('loss', 'N/A'):<10}"
        precision_value = f"{precision:<10.4f}" if isinstance(precision, float) else f"{precision:<10}"
        recall_value = f"{recall:<10.4f}" if isinstance(recall, float) else f"{recall:<10}"
        ndcg_value = f"{metrics.get('ndcg', 'N/A'):<10.4f}" if isinstance(metrics.get('ndcg'), float) else f"{metrics.get('ndcg', 'N/A'):<10}"

        print(f"{'Value':<10} {loss_value} {precision_value} {recall_value} {ndcg_value}")
        print("-" * 50)

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        raise

# Example Usage
if __name__ == "__main__":
    evaluate()