# import configparser
# import os
# import torch
# from src.preprocessing.data_loader import load_reviews_sample, load_meta, clean_reviews
# from src.preprocessing.graph_builder import build_graph
# from src.datasets.LinkPredictionDataset import LinkPredictionDataset
# from src.models.graphsage import GraphSAGEModel
# from src.training.train import train
# from src.training.eval import evaluate_while_training, evaluate_link_prediction
# from src.training.eval import evaluate_classification_metrics, evaluate_ranking_metrics
# from src.training.save_trained_model import save_trained_model
# from src.inference.inference import inference
# from src.recommender.recommender import recommender

# # Read configuration
# config = configparser.ConfigParser()
# config.read('./config/base_config.ini')

# # Constants
# REVIEWS_DATA_PATH = "./data/raw/Electronics_5.csv"
# METADATA_DATA_PATH = "./data/raw/meta_Electronics.jsonl"
# PROCESSED_DATA_PATH = "./data/processed"
# SAMPLE_SIZE = int(config['SAMPLING']['SAMPLE_SIZE'])
# CHUNK_SIZE = int(config['SAMPLING']['CHUNK_SIZE'])
# EPOCHS_SIZE = int(config['MODEL']['EPOCHS']) + 1

# def load_data():
#     """Load and clean reviews and metadata."""
#     reviews_df = load_reviews_sample(REVIEWS_DATA_PATH, sample_size=SAMPLE_SIZE, chunk_size=CHUNK_SIZE)
#     asins = reviews_df['parent_asin'].unique().tolist()
#     meta_df = load_meta(METADATA_DATA_PATH, asins_to_keep=asins)
#     reviews_df = clean_reviews(reviews_df)
#     return reviews_df, meta_df

# def save_processed_data(edge_index, features, labels=None):
#     """Save the preprocessed graph data to files."""
#     os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
#     torch.save(edge_index, os.path.join(PROCESSED_DATA_PATH, "edge_index.pt"))
#     torch.save(features, os.path.join(PROCESSED_DATA_PATH, "node_features.pt"))
    
#     if labels is not None:
#         torch.save(labels, os.path.join(PROCESSED_DATA_PATH, "labels.pt"))

#     print("Data saved to processed directory.")

# def load_processed_data():
#     """Load the preprocessed graph data from files."""
#     edge_index = torch.load(os.path.join(PROCESSED_DATA_PATH, "edge_index.pt"))
#     features = torch.load(os.path.join(PROCESSED_DATA_PATH, "node_features.pt"))
#     # Optionally, load labels if available
#     labels = None
#     # Uncomment if labels exist
#     # labels = torch.load(os.path.join(PROCESSED_DATA_PATH, "labels.pt"))
#     return edge_index, features, labels

# def preprocess():
#     """Main preprocessing pipeline."""
#     print(f"Loading reviews from: {REVIEWS_DATA_PATH}")
#     print(f"Sample Size: {SAMPLE_SIZE}")

#     reviews_df, meta_df = load_data()
#     data = build_graph(reviews_df, meta_df)
    
#     edge_index = data.edge_index
#     features = data.x
#     labels = None  # Set if labels exist in your data

#     save_processed_data(edge_index, features, labels)
    
#     # Verify the saved data
#     edge_index, features, labels = load_processed_data()
#     print(f"Edge Index: {edge_index.shape}")
#     print(f"Features: {features.shape}")
#     if labels is not None:
#         print(f"Labels: {labels.shape}")

# def train_model():
#     """Train the GraphSAGE model."""
#     edge_index, features, _ = load_processed_data()

#     # Prepare dataset
#     dataset = LinkPredictionDataset(edge_index, num_nodes=features.size(0))
#     data = dataset.get_data(features)
#     train_pos, train_neg = dataset.train_edges, dataset.train_neg_edges
#     test_pos, test_neg = dataset.test_edges, dataset.test_neg_edges

#     # Initialize the GraphSAGE model and optimizer
#     model = GraphSAGEModel(in_channels=features.size(1), hidden_channels=64, out_channels=32)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     # Training loop
#     for epoch in range(1, EPOCHS_SIZE):
#         loss = train(model, data, optimizer, train_pos, train_neg)
        
#         if epoch % 10 == 0:
#             acc, auc, precision, recall, f1, _ = evaluate_while_training(model, data, test_pos, test_neg)
            
#             model.eval()
#             with torch.no_grad():
#                 z = model(data.x, data.edge_index)
#                 mrr, hits = evaluate_link_prediction(z, test_pos, test_neg, k=10)

#             print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | AUC: {auc:.4f} | "
#                   f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | "
#                   f"MRR: {mrr:.4f} | Hits@10: {hits:.4f}")

#     # Save the trained model and embeddings
#     model.eval()
#     with torch.no_grad():
#         z = model(data.x, data.edge_index)

#     MODEL_PATH = "./saved_models/graphsage_model.pt"
#     EMBED_PATH = "./saved_models/node_embeddings.pt"
#     save_trained_model(model, z, MODEL_PATH, EMBED_PATH)

# def inference_mode():
#     edge_index = torch.load("./data/processed/edge_index.pt")
#     node_features = torch.load("./data/processed/node_features.pt")

#     model = GraphSAGEModel(in_channels=node_features.size(1), hidden_channels=64, out_channels=32)
#     model.load_state_dict(torch.load("./saved_models/graphsage_model.pt"))
#     model.eval()

#     dataset = LinkPredictionDataset(edge_index, num_nodes=node_features.size(0))
#     data = dataset.get_data(node_features)
    
#     test_pos, test_neg = dataset.test_edges, dataset.test_neg_edges

#     pos_probs = inference(model, data, test_pos)
#     neg_probs = inference(model, data, test_neg)

#     all_probs = torch.cat([pos_probs, neg_probs])
#     print("Predicted probabilities for test edges:")
#     print(all_probs)

#     user_id = 42  # or loop over many users
#     recommendations_amount = 10
#     top_items, scores = recommender(user_id, node_features, edge_index, dataset.train_edges, top_k=recommendations_amount)

#     print(f"\nTop recommended items for user {user_id}:")
#     for idx, score in zip(top_items.tolist(), scores.tolist()):
#         print(f"Item {idx} with score {score:.4f}")

# def evaluate_model():
#     """Evaluate the trained GraphSAGE model on test data."""
#     print("\n🔍 Starting Evaluation Phase...")

#     # Load processed data
#     edge_index = torch.load(os.path.join(PROCESSED_DATA_PATH, "edge_index.pt"))
#     features = torch.load(os.path.join(PROCESSED_DATA_PATH, "node_features.pt"))

#     # Load dataset and split edges
#     dataset = LinkPredictionDataset(edge_index, num_nodes=features.size(0))
#     data = dataset.get_data(features)
#     test_pos_edge = dataset.test_edges
#     test_neg_edge = dataset.test_neg_edges

#     # Load model
#     model_path = "./saved_models/graphsage_model.pt"
#     embed_path = "./saved_models/node_embeddings.pt"

#     model = GraphSAGEModel(in_channels=features.size(1), hidden_channels=64, out_channels=32)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     # Load or recompute embeddings
#     if os.path.exists(embed_path):
#         z = torch.load(embed_path)
#     else:
#         with torch.no_grad():
#             z = model(data.x, data.edge_index)

#     # Evaluate metrics
    

#     acc, auc, precision, recall, f1 = evaluate_classification_metrics(z, test_pos_edge, test_neg_edge)
#     mrr, hits_at_10 = evaluate_ranking_metrics(z, test_pos_edge, test_neg_edge, k=10)

#     print("\n📊 Model Evaluation Metrics:")
#     print(f"✅ Accuracy:  {acc:.4f}")
#     print(f"✅ AUC:       {auc:.4f}")
#     print(f"✅ Precision: {precision:.4f}")
#     print(f"✅ Recall:    {recall:.4f}")
#     print(f"✅ F1 Score:  {f1:.4f}")
#     print(f"⭐ MRR:       {mrr:.4f}")
#     print(f"⭐ Hits@10:   {hits_at_10:.4f}")


# def main():
#     """Main function to run the entire workflow."""
#     preprocess()
#     train_model()
#     evaluate_model()
#     # inference_mode()

# if __name__ == "__main__":
#     main()

import configparser
import os
import torch
import logging

# Import functions from our internal modules
from src.preprocessing.data_loader import load_reviews_sample, load_meta, clean_reviews
from src.preprocessing.graph_builder import build_graph
from src.datasets.LinkPredictionDataset import LinkPredictionDataset
from src.models.graphsage import GraphSAGEModel
from src.training.train import train
from src.training.eval import (
    evaluate_while_training, 
    evaluate_link_prediction, 
    evaluate_classification_metrics, 
    evaluate_ranking_metrics
)
from src.training.save_trained_model import save_trained_model
from src.inference.inference import inference
from src.recommender.recommender import recommender

# Set up logging for the overall pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Read configuration from file
config = configparser.ConfigParser()
config.read('./config/base_config.ini')

# Constants extracted from configuration and fixed paths
REVIEWS_DATA_PATH = "./data/raw/Electronics_5.csv"
METADATA_DATA_PATH = "./data/raw/meta_Electronics.jsonl"
PROCESSED_DATA_PATH = "./data/processed"
MODEL_SAVE_DIR = "./saved_models"
SAMPLE_SIZE = int(config['SAMPLING']['SAMPLE_SIZE'])
CHUNK_SIZE = int(config['SAMPLING']['CHUNK_SIZE'])
EPOCHS = int(config['MODEL'].get('EPOCHS', 1000))  # Default to 1000 epochs if not set

def load_data() -> tuple:
    """
    Load and clean reviews and metadata.

    Returns:
        tuple: (reviews_df, meta_df)
    """
    logging.info(f"Loading reviews from {REVIEWS_DATA_PATH} (Sample Size: {SAMPLE_SIZE}, Chunk Size: {CHUNK_SIZE})")
    reviews_df = load_reviews_sample(REVIEWS_DATA_PATH, sample_size=SAMPLE_SIZE, chunk_size=CHUNK_SIZE)
    asins = reviews_df['parent_asin'].unique().tolist()
    meta_df = load_meta(METADATA_DATA_PATH, asins_to_keep=asins)
    reviews_df = clean_reviews(reviews_df)
    return reviews_df, meta_df

def save_processed_data(edge_index: torch.Tensor, features: torch.Tensor, labels: torch.Tensor = None) -> None:
    """
    Save the preprocessed graph data to disk.

    Args:
        edge_index (torch.Tensor): Graph edge indices.
        features (torch.Tensor): Node features.
        labels (torch.Tensor, optional): Graph labels if available.
    """
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    torch.save(edge_index, os.path.join(PROCESSED_DATA_PATH, "edge_index.pt"))
    torch.save(features, os.path.join(PROCESSED_DATA_PATH, "node_features.pt"))
    if labels is not None:
        torch.save(labels, os.path.join(PROCESSED_DATA_PATH, "labels.pt"))
    logging.info("Data saved to processed directory.")

def load_processed_data() -> tuple:
    """
    Load the preprocessed graph data from disk.

    Returns:
        tuple: (edge_index, features, labels)
    """
    edge_index = torch.load(os.path.join(PROCESSED_DATA_PATH, "edge_index.pt"))
    features = torch.load(os.path.join(PROCESSED_DATA_PATH, "node_features.pt"))
    labels = None  # Uncomment if you save labels: torch.load(os.path.join(PROCESSED_DATA_PATH, "labels.pt"))
    return edge_index, features, labels

def preprocess() -> None:
    """
    Run the preprocessing pipeline:
      1. Load reviews and metadata.
      2. Build the graph.
      3. Save processed graph data.
    """
    logging.info("=== Preprocessing Phase ===")
    reviews_df, meta_df = load_data()
    data = build_graph(reviews_df, meta_df)
    
    edge_index = data.edge_index
    features = data.x
    labels = None  # Update if your data includes labels

    save_processed_data(edge_index, features, labels)
    
    # Verification step
    edge_index, features, labels = load_processed_data()
    logging.info(f"Edge Index Shape: {edge_index.shape}")
    logging.info(f"Node Features Shape: {features.shape}")
    if labels is not None:
        logging.info(f"Labels Shape: {labels.shape}")

def train_model() -> None:
    """
    Train the GraphSAGE model using processed data.
    """
    logging.info("=== Training Phase ===")
    edge_index, features, _ = load_processed_data()

    # Prepare the dataset
    dataset = LinkPredictionDataset(edge_index, num_nodes=features.size(0))
    data = dataset.get_data(features)
    train_pos, train_neg = dataset.train_edges, dataset.train_neg_edges
    test_pos, test_neg = dataset.test_edges, dataset.test_neg_edges

    # Initialize model and optimizer
    model = GraphSAGEModel(in_channels=features.size(1), hidden_channels=64, out_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, data, optimizer, train_pos, train_neg)
        if epoch % 10 == 0:
            # Evaluate during training
            acc, auc, precision, recall, f1, _ = evaluate_while_training(model, data, test_pos, test_neg)
            with torch.no_grad():
                z = model(data.x, data.edge_index)
                mrr, hits = evaluate_link_prediction(z, test_pos, test_neg, k=10)
            logging.info(
                f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | AUC: {auc:.4f} | "
                f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | "
                f"MRR: {mrr:.4f} | Hits@10: {hits:.4f}"
            )

    # Save final model and embeddings
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "graphsage_model.pt")
    EMBED_PATH = os.path.join(MODEL_SAVE_DIR, "node_embeddings.pt")
    save_trained_model(model, z, MODEL_PATH, EMBED_PATH)
    logging.info("Training completed and model saved.")

def evaluate_model() -> dict:
    """
    Evaluate the trained GraphSAGE model on the test set.
    
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    logging.info("=== Evaluation Phase ===")
    edge_index, features, _ = load_processed_data()
    dataset = LinkPredictionDataset(edge_index, num_nodes=features.size(0))
    data = dataset.get_data(features)
    test_pos_edge = dataset.test_edges
    test_neg_edge = dataset.test_neg_edges

    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "graphsage_model.pt")
    EMBED_PATH = os.path.join(MODEL_SAVE_DIR, "node_embeddings.pt")

    model = GraphSAGEModel(in_channels=features.size(1), hidden_channels=64, out_channels=32)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    if os.path.exists(EMBED_PATH):
        z = torch.load(EMBED_PATH)
    else:
        with torch.no_grad():
            z = model(data.x, data.edge_index)

    acc, auc, precision, recall, f1 = evaluate_classification_metrics(z, test_pos_edge, test_neg_edge)
    mrr, hits_at_10 = evaluate_ranking_metrics(z, test_pos_edge, test_neg_edge, k=10)

    logging.info("📊 Model Evaluation Metrics:")
    logging.info(f"Accuracy:  {acc:.4f}")
    logging.info(f"AUC:       {auc:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1 Score:  {f1:.4f}")
    logging.info(f"MRR:       {mrr:.4f}")
    logging.info(f"Hits@10:   {hits_at_10:.4f}")

    return {
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "hits@10": hits_at_10
    }

def inference_mode() -> None:
    """
    Run inference and recommendation for a sample user.
    """
    logging.info("=== Inference Phase ===")
    edge_index = torch.load(os.path.join(PROCESSED_DATA_PATH, "edge_index.pt"))
    node_features = torch.load(os.path.join(PROCESSED_DATA_PATH, "node_features.pt"))

    model = GraphSAGEModel(in_channels=node_features.size(1), hidden_channels=64, out_channels=32)
    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "graphsage_model.pt")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    dataset = LinkPredictionDataset(edge_index, num_nodes=node_features.size(0))
    data = dataset.get_data(node_features)
    test_pos, test_neg = dataset.test_edges, dataset.test_neg_edges

    # Run inference to obtain predicted probabilities for test edges.
    pos_probs = inference(model, data, test_pos)
    neg_probs = inference(model, data, test_neg)
    all_probs = torch.cat([pos_probs, neg_probs])
    logging.info("Predicted probabilities for test edges:")
    logging.info(all_probs)

    # Retrieve recommendations for a sample user (e.g., user with ID 42)
    user_id = 42
    top_items, scores = recommender(user_id, node_features, edge_index, dataset.train_edges, top_k=10)

    logging.info(f"Top recommended items for user {user_id}:")
    for idx, score in zip(top_items.tolist(), scores.tolist()):
        logging.info(f"Item {idx} with score {score:.4f}")

def main() -> None:
    """Main function to run the full workflow: preprocessing, training, evaluation, and inference."""
    # Uncomment the phases you want to run:
    # preprocess()
    # train_model()
    # evaluate_model()
    inference_mode()

if __name__ == "__main__":
    main()
