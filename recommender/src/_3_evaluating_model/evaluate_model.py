# import os
# import torch
# import torch.nn.functional as F
# from torch_geometric.loader import DataLoader  
# from typing import Dict

# from src._1_model_selection.GraphSAGEModelV0 import GraphSAGEModelV0
# import torch_geometric
# from settings import PROCESSED_DATA_DIR, CHECKPOINT_DIR, MODEL_NAME, ENABLE_CUDA, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, NUM_LAYERS, DROPOUT_RATE

# def link_prediction_loss(pos_edge_index, neg_edge_index, z):
#     # pos_edge_index: positive edge indices (real edges in the graph)
#     # neg_edge_index: negative edge indices (non-existing edges)
    
#     # Get node embeddings for positive and negative edges
#     pos_edge_emb = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
#     neg_edge_emb = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    
#     # Positive edge score (the higher, the more likely it is a real edge)
#     pos_loss = F.logsigmoid(pos_edge_emb).mean()
    
#     # Negative edge score (the lower, the more likely it is a fake edge)
#     neg_loss = F.logsigmoid(-neg_edge_emb).mean()
    
#     return -(pos_loss + neg_loss)


# def precision_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
#     """
#     Compute precision@k for the given scores and labels.
#     """
#     _, indices = torch.topk(scores, k)
#     top_k_labels = labels[indices]
#     precision = top_k_labels.sum().item() / k
#     return precision


# def recall_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
#     """
#     Compute recall@k for the given scores and labels.
#     """
#     _, indices = torch.topk(scores, k)
#     top_k_labels = labels[indices]
#     recall = top_k_labels.sum().item() / labels.sum().item()
#     return recall


# def ndcg_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
#     """
#     Compute Normalized Discounted Cumulative Gain (NDCG) for the given scores and labels.
#     """
#     _, indices = torch.sort(scores, descending=True)
#     sorted_labels = labels[indices]
#     dcg = (sorted_labels / torch.log2(torch.arange(len(sorted_labels), dtype=torch.float) + 2)).sum().item()
#     ideal_labels, _ = torch.sort(labels, descending=True)
#     idcg = (ideal_labels / torch.log2(torch.arange(len(ideal_labels), dtype=torch.float) + 2)).sum().item()
#     return dcg / idcg if idcg > 0 else 0.0





# def evaluate():
#     """
#     Function to evaluate the trained model.
#     Includes steps like model evaluation on test data and performance metrics calculation.
#     """
#     print("Evaluating the model...")

#     # Step 1: Load test data
#     try:
#         test_data_path = os.path.join(PROCESSED_DATA_DIR, "graph/positives/test_data.pt")
#         test_neg_samples_path = os.path.join(PROCESSED_DATA_DIR, "graph/negatives/test_neg_samples.pt")

#         if not os.path.exists(test_data_path):
#             raise FileNotFoundError(f"Test data file not found at {test_data_path}")
#         if not os.path.exists(test_neg_samples_path):
#             raise FileNotFoundError(f"Negative samples file not found at {test_neg_samples_path}")

#         test_data = torch.load(test_data_path, weights_only=False)
#         test_neg_samples = torch.load(test_neg_samples_path, weights_only=False)

#         # Debugging: Print the type and structure of loaded data
#         print(f"Type of test_data: {type(test_data)}")
#         print(f"Type of test_neg_samples: {type(test_neg_samples)}")

#         # Ensure the loaded data is of the correct type
#         if not isinstance(test_data, torch_geometric.data.Data):
#             raise TypeError("test_data must be an instance of torch_geometric.data.Data")
#         if not isinstance(test_neg_samples, torch_geometric.data.Data):
#             # Attempt to convert test_neg_samples to Data if it's a dictionary
#             if isinstance(test_neg_samples, dict):
#                 test_neg_samples = torch_geometric.data.Data(
#                     x=test_neg_samples.get("x", None),
#                     edge_index=test_neg_samples.get("edge_index", None),
#                     y=test_neg_samples.get("y", None),
#                 )
#             else:
#                 raise TypeError("test_neg_samples must be an instance of torch_geometric.data.Data")

#     except Exception as e:
#         print(f"Error loading test data: {e}")
#         raise

#     # Step 2: Load the trained model
#     model_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pt")
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

#     try:
#         # Load the checkpoint (contains state_dict and metadata)
#         checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

#         # Initialize the model architecture
#         input_dim = test_data.num_node_features  # Get input dimensions from test data
#         model = GraphSAGEModelV0(
#             input_dim=input_dim,
#             hidden_dim=HIDDEN_DIM,
#             output_dim=OUTPUT_DIM,
#             num_layers=NUM_LAYERS,
#             dropout=DROPOUT_RATE,
#         )

#         # Load the model weights from the checkpoint
#         model.load_state_dict(checkpoint['model_state_dict'])

#     except Exception as e:
#         print(f"Error loading the model: {e}")
#         raise

#     # Move the model to the appropriate device
#     device = "cuda" if ENABLE_CUDA and torch.cuda.is_available() else "cpu"
#     model.to(device)

#     # Step 3: Evaluate the model
#     evaluate_testing_model(
#         model=model,
#         test_data=test_data,
#         test_neg_samples=test_neg_samples,
#         device=device,
#     )

# def evaluate_testing_model(
#     model: torch.nn.Module,
#     test_data: torch.Tensor,
#     test_neg_samples: torch.Tensor,
#     device: str,
# ) -> None:
#     """
#     Evaluate the model on the test dataset.
#     This function calculates ranking metrics like precision@k, recall@k, and NDCG.
#     """
#     if not isinstance(model, torch.nn.Module):
#         raise TypeError("model must be an instance of torch.nn.Module")
#     if not isinstance(test_data, torch_geometric.data.Data):
#         raise TypeError("test_data must be an instance of torch_geometric.data.Data")
#     if not isinstance(test_neg_samples, torch_geometric.data.Data):
#         raise TypeError("test_neg_samples must be an instance of torch_geometric.data.Data")

#     model.eval()
#     all_scores, all_labels = [], []

#     with torch.no_grad():
#         combined_data = torch.cat([test_data, test_neg_samples], dim=0).to(device)
#         labels = torch.cat([torch.ones(len(test_data)), torch.zeros(len(test_neg_samples))]).to(device)

#         output = model(combined_data.x, combined_data.edge_index)

#         src_embeddings = F.normalize(output[combined_data.edge_index[0]], p=2, dim=1)
#         dst_embeddings = F.normalize(output[combined_data.edge_index[1]], p=2, dim=1)
#         edge_scores = (src_embeddings * dst_embeddings).sum(dim=1)

#         all_scores.append(edge_scores.cpu())
#         all_labels.append(labels.cpu())

#     all_scores = torch.cat(all_scores)
#     all_labels = torch.cat(all_labels)

#     precision = precision_at_k(all_scores, all_labels, k=10)
#     recall = recall_at_k(all_scores, all_labels, k=10)
#     ndcg = ndcg_score(all_scores, all_labels)

#     print("Evaluation Results:")
#     print(f"Precision@10: {precision:.4f}")
#     print(f"Recall@10: {recall:.4f}")
#     print(f"NDCG: {ndcg:.4f}")



# import torch
# from sklearn.metrics import precision_score, recall_score, ndcg_score
# from torch_geometric.loader import DataLoader

# def test_model(
#     model: torch.nn.Module,
#     test_dataloader: DataLoader,
#     loss_fn: torch.nn.Module,
#     device: str = "cpu"
# ):
#     """
#     Evaluate the trained model on the test dataset.

#     Args:
#         model (torch.nn.Module): The trained model.
#         test_dataloader (DataLoader): DataLoader for the test dataset.
#         loss_fn (torch.nn.Module): Loss function used for evaluation.
#         device (str): Device to run the evaluation on ('cpu' or 'cuda').

#     Returns:
#         dict: A dictionary containing evaluation metrics (loss, precision, recall, NDCG).
#     """
#     print("Starting model evaluation on test data...")

#     # Set the model to evaluation mode
#     model.eval()
#     model.to(device)

#     total_loss = 0.0
#     all_labels = []
#     all_predictions = []

#     with torch.no_grad():  # Disable gradient computation
#         for data in test_dataloader:
#             data = data.to(device)

#             # Forward pass: Get predictions
#             output = model(data.x, data.edge_index)

#             # Compute edge scores (dot product between connected nodes)
#             src_embeddings = output[data.edge_index[0]]  # Source node embeddings
#             dst_embeddings = output[data.edge_index[1]]  # Destination node embeddings
#             edge_scores = (src_embeddings * dst_embeddings).sum(dim=1)  # Dot product for edge scores

#             # Compute loss
#             loss = loss_fn(edge_scores, data.y.float())
#             total_loss += loss.item()

#             # Collect labels and predictions
#             all_labels.append(data.y.cpu().numpy())  # Ground truth labels
#             all_predictions.append(torch.sigmoid(edge_scores).cpu().numpy())  # Predicted probabilities

#     # Concatenate all labels and predictions
#     all_labels = np.concatenate(all_labels)
#     all_predictions = np.concatenate(all_predictions)

#     # Convert predictions to binary labels (threshold = 0.5)
#     binary_predictions = (all_predictions > 0.5).astype(int)

#     # Compute evaluation metrics
#     precision = precision_score(all_labels, binary_predictions)
#     recall = recall_score(all_labels, binary_predictions)
#     ndcg = ndcg_score([all_labels], [all_predictions])  # NDCG requires lists of lists

#     # Average loss over all batches
#     avg_loss = total_loss / len(test_dataloader)

#     # Print evaluation results
#     print(f"Test Loss: {avg_loss:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"NDCG: {ndcg:.4f}")

#     # Return metrics as a dictionary
#     return {
#         "loss": avg_loss,
#         "precision": precision,
#         "recall": recall,
#         "ndcg": ndcg,
#     }

# def evaluate_test_model(
#     test_data_path: str,
#     checkpoint_path: str,
#     hidden_dim: int,
#     num_layers: int,
#     dropout_rate: float,
#     batch_size: int,
#     device: str = "cpu"
# ):
#     """
#     Evaluate the trained model on the test dataset.

#     Args:
#         test_data_path (str): Path to the test dataset file.
#         checkpoint_path (str): Path to the saved model checkpoint.
#         hidden_dim (int): Hidden dimension size for the model.
#         num_layers (int): Number of layers in the model.
#         dropout_rate (float): Dropout rate for the model.
#         batch_size (int): Batch size for the DataLoader.
#         device (str): Device to run the evaluation on ('cpu' or 'cuda').

#     Returns:
#         dict: A dictionary containing evaluation metrics (loss, precision, recall, NDCG).
#     """
#     print("Starting model evaluation...")

#     # Load the test dataset
#     test_dataset = GraphDataset(test_data_path)
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

#     # Initialize the model
#     input_dim = test_dataset.num_node_features
#     model = GraphSAGEModelV0(
#         input_dim=input_dim,
#         hidden_dim=hidden_dim,
#         output_dim=1,  # Assuming binary classification (output_dim=1)
#         num_layers=num_layers,
#         dropout=dropout_rate,
#     ).to(device)

#     # Load the trained model weights
#     try:
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print("Model weights loaded successfully.")
#     except Exception as e:
#         print(f"[ERROR] Failed to load model weights: {e}")
#         raise

#     # Define the loss function
#     loss_fn = torch.nn.BCEWithLogitsLoss()

#     # Test the model
#     metrics = test_model(
#         model=model,
#         test_dataloader=test_dataloader,
#         loss_fn=loss_fn,
#         device=device
#     )

#     # Print metrics
#     print("Evaluation Metrics:")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}")

#     return metrics


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
):
    """
    Evaluate the trained model on test data and compute ranking metrics.
    
    Args:
        model (torch.nn.Module): Trained model.
        test_data (Data): Test dataset containing positive edges.
        test_neg_samples (Data): Test dataset containing negative edges.
        loss_fn (Optional[torch.nn.Module]): Loss function (optional).
        device (str): Device to run evaluation on ('cpu' or 'cuda').
    
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    if not isinstance(test_data, Data):
        raise TypeError("test_data must be an instance of torch_geometric.data.Data")
    if not isinstance(test_neg_samples, Data):
        raise TypeError("test_neg_samples must be an instance of torch_geometric.data.Data")

    # Move model and data to the specified device
    model.eval()
    model.to(device)
    test_data = test_data.to(device)
    test_neg_samples = test_neg_samples.to(device)

    all_scores, all_labels = [], []

    with torch.no_grad():
        # Get node embeddings
        output = model(test_data.x, test_data.edge_index)

        # Compute scores for positive edges
        src_embeddings = output[test_data.edge_index[0]]
        dst_embeddings = output[test_data.edge_index[1]]
        pos_scores = (src_embeddings * dst_embeddings).sum(dim=1)

        # Compute scores for negative edges
        src_embeddings_neg = output[test_neg_samples.edge_index[0]]
        dst_embeddings_neg = output[test_neg_samples.edge_index[1]]
        neg_scores = (src_embeddings_neg * dst_embeddings_neg).sum(dim=1)

        # Combine scores and labels
        all_scores.append(torch.cat([pos_scores.cpu(), neg_scores.cpu()]))
        all_labels.append(torch.cat([torch.ones_like(pos_scores.cpu()), torch.zeros_like(neg_scores.cpu())]))

    # Concatenate scores and labels
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    # Compute ranking metrics
    precision = precision_at_k(all_scores, all_labels, k=10)
    recall = recall_at_k(all_scores, all_labels, k=10)
    ndcg = ndcg_score(all_scores, all_labels)

    # Compute loss if provided
    loss = None
    if loss_fn:
        loss = loss_fn(all_scores, all_labels.float())

    # Print metrics
    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10: {recall:.4f}")
    print(f"NDCG: {ndcg:.4f}")
    if loss:
        print(f"Loss: {loss.item():.4f}")

    return {
        "precision@10": precision,
        "recall@10": recall,
        "ndcg": ndcg,
        "loss": loss.item() if loss else None,
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