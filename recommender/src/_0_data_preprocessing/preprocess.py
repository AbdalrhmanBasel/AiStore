import os
import sys
import torch

from colorama import Fore, Style, init
import torch_geometric

init()

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)

# Corrected imports
from src._0_data_preprocessing.utils.data import load_data
from src._0_data_preprocessing.utils.clean_reviews import clean_reviews
from src._0_data_preprocessing.utils.clean_meta import clean_meta
from src._0_data_preprocessing.utils.features_matrix import (
    create_feature_matrix,
    feature_engineering,
    normalize_features,
)
from src._0_data_preprocessing.utils.graphs import construct_graph, save_graph
from src._0_data_preprocessing.utils.split_data import split_edges, add_negative_samples, report_graph_details

# Import settings and utilities
from settings import (
    REVIEW_DATA_PATH,
    META_DATA_PATH,
    SAMPLE_META_DATA_PATH,
    SAMPLE_REVIEW_DATA_PATH,
    SAMPLE_CLEANED_META_DATA_PATH,
    SAMPLE_CLEANED_REVIEW_DATA_PATH,
    FEATURES_MATRIX_PATH,
    GRAPH_SAVE_PATH,
    PROCESSED_DATA_DIR,
)

def load_and_clean_data():
    """
    Load raw reviews and metadata, then clean and save samples.
    """
    print(f"{Fore.CYAN}[INFO] Loading raw reviews and metadata...{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}🔄 Sampling 1000 reviews from '{REVIEW_DATA_PATH}' in chunks of 1000...{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}🔄 Loading metadata from '{META_DATA_PATH}' in chunks of 1000...{Style.RESET_ALL}")

    try:
        reviews_df, meta_df = load_data(REVIEW_DATA_PATH, META_DATA_PATH)
        print(f"{Fore.GREEN}✅ Successfully loaded {reviews_df.shape[0]} reviews.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✅ Loaded {meta_df.shape[0]} filtered metadata records.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to load data: {e}{Style.RESET_ALL}")
        raise

    print(f"{Fore.CYAN}[INFO] Cleaning datasets...{Style.RESET_ALL}")
    cleaned_reviews = clean_reviews(reviews_df)
    cleaned_meta = clean_meta(meta_df)

    print(f"{Fore.CYAN}[INFO] Saving dataset samples and cleaned versions...{Style.RESET_ALL}")
    os.makedirs(os.path.dirname(SAMPLE_REVIEW_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SAMPLE_META_DATA_PATH), exist_ok=True)

    try:
        reviews_df.to_csv(SAMPLE_REVIEW_DATA_PATH, index=False)
        meta_df.to_json(SAMPLE_META_DATA_PATH, orient="records", lines=True)
        cleaned_reviews.to_csv(SAMPLE_CLEANED_REVIEW_DATA_PATH, index=False)
        cleaned_meta.to_csv(SAMPLE_CLEANED_META_DATA_PATH, index=False)

        print(f"{Fore.GREEN}✅ Saved raw samples: {SAMPLE_REVIEW_DATA_PATH}, {SAMPLE_META_DATA_PATH}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✅ Saved cleaned datasets: {SAMPLE_CLEANED_REVIEW_DATA_PATH}, {SAMPLE_CLEANED_META_DATA_PATH}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to save datasets: {e}{Style.RESET_ALL}")
        raise

    print(f"{Fore.BLUE}[INFO] Raw Shapes — Reviews: {reviews_df.shape}, Meta: {meta_df.shape}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}[INFO] Cleaned Shapes — Reviews: {cleaned_reviews.shape}, Meta: {cleaned_meta.shape}{Style.RESET_ALL}")

    return cleaned_reviews, cleaned_meta


def generate_feature_matrix(cleaned_meta):
    """
    Generate and save the feature matrix from cleaned metadata.
    """
    print(f"\n{Fore.CYAN}[INFO] Generating feature matrix...{Style.RESET_ALL}")
    try:
        feature_matrix = create_feature_matrix(cleaned_meta)
        feature_matrix = feature_engineering(feature_matrix)
        feature_matrix = normalize_features(feature_matrix)

        os.makedirs(os.path.dirname(FEATURES_MATRIX_PATH), exist_ok=True)
        feature_matrix.to_csv(FEATURES_MATRIX_PATH, index=False)

        print(f"{Fore.GREEN}✅ Feature matrix saved to: {FEATURES_MATRIX_PATH}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to generate or save feature matrix: {e}{Style.RESET_ALL}")
        raise

    return feature_matrix


def build_graph(cleaned_meta, feature_matrix, reviews_df):
    """
    Construct and save the product-user interaction graph.
    """
    print(f"\n{Fore.CYAN}[INFO] Building product-user interaction graph...{Style.RESET_ALL}")

    node_features = [
        "price",
        "average_rating",
        "rating_number",
        "num_features",
        "description_length",
        "price_per_rating",
        "feature_to_rating_ratio",
    ]

    # Ensure all required features are present in the feature matrix
    missing_features = [f for f in node_features if f not in feature_matrix.columns]
    if missing_features:
        print(f"{Fore.RED}[ERROR] Missing required features in feature matrix: {missing_features}{Style.RESET_ALL}")
        raise ValueError("Missing required features in feature matrix.")

    try:
        graph_data = construct_graph(
            meta_df=cleaned_meta,
            meta_features_df=feature_matrix,
            reviews_df=reviews_df,
            product_col="parent_asin",
            user_col="user_id",
            label_col=None,  # or set if you're doing classification
        )

        report_graph_details(graph_data)

        os.makedirs(os.path.dirname(GRAPH_SAVE_PATH), exist_ok=True)
        save_graph(graph_data, GRAPH_SAVE_PATH)

        print(f"{Fore.GREEN}✅ Graph saved to: {GRAPH_SAVE_PATH}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to build or save graph: {e}{Style.RESET_ALL}")
        raise

    return graph_data


# def split_and_save_data(graph_data):
#     """
#     Split graph data into training/validation/test sets and generate negative samples.
#     """
#     print(f"\n{Fore.CYAN}[INFO] Splitting data into training, validation, and test sets...{Style.RESET_ALL}")

#     try:
#         # Split edges into training, validation, and test sets
#         train_edges, val_edges, test_edges = split_edges(graph_data)

#         # Generate negative samples for each split
#         train_neg_samples = add_negative_samples(train_edges, graph_data)
#         val_neg_samples = add_negative_samples(val_edges, graph_data)
#         test_neg_samples = add_negative_samples(test_edges, graph_data)

#         # Reconstruct graph objects for each split
#         train_graph = torch_geometric.data.Data(
#             x=graph_data.x,  # Use the same node features as the original graph
#             edge_index=train_edges,  # Use the split edges
#             num_nodes=graph_data.num_nodes  # Preserve the number of nodes
#         )
#         val_graph = torch_geometric.data.Data(
#             x=graph_data.x,
#             edge_index=val_edges,
#             num_nodes=graph_data.num_nodes
#         )
#         test_graph = torch_geometric.data.Data(
#             x=graph_data.x,
#             edge_index=test_edges,
#             num_nodes=graph_data.num_nodes
#         )

#         # Create directories for saving split data
#         os.makedirs(os.path.join(PROCESSED_DATA_DIR, "graph/positives"), exist_ok=True)
#         os.makedirs(os.path.join(PROCESSED_DATA_DIR, "graph/negatives"), exist_ok=True)

#         # Save positive edges
#         train_data_path = os.path.join(PROCESSED_DATA_DIR, "graph/positives/train_data.pt")
#         val_data_path = os.path.join(PROCESSED_DATA_DIR, "graph/positives/val_data.pt")
#         test_data_path = os.path.join(PROCESSED_DATA_DIR, "graph/positives/test_data.pt")

#         torch.save(train_graph, train_data_path)
#         torch.save(val_graph, val_data_path)
#         torch.save(test_graph, test_data_path)

#         # Save negative samples
#         train_neg_samples_path = os.path.join(PROCESSED_DATA_DIR, "graph/negatives/train_neg_samples.pt")
#         val_neg_samples_path = os.path.join(PROCESSED_DATA_DIR, "graph/negatives/val_neg_samples.pt")
#         test_neg_samples_path = os.path.join(PROCESSED_DATA_DIR, "graph/negatives/test_neg_samples.pt")

#         torch.save(train_neg_samples, train_neg_samples_path)
#         torch.save(val_neg_samples, val_neg_samples_path)
#         torch.save(test_neg_samples, test_neg_samples_path)

#         print(f"{Fore.GREEN}✅ Saved split data: {train_data_path}, {val_data_path}, {test_data_path}{Style.RESET_ALL}")
#         print(f"{Fore.GREEN}✅ Saved negative samples data for training, validation, and testing.{Style.RESET_ALL}")
#     except Exception as e:
#         print(f"{Fore.RED}[ERROR] Failed to split or save data: {e}{Style.RESET_ALL}")
#         raise

#     # Report details of each split
#     print(f"\n{Fore.CYAN}[INFO] Reporting details of training graph...{Style.RESET_ALL}")
#     report_graph_details(train_graph)

#     print(f"\n{Fore.CYAN}[INFO] Reporting details of validation graph...{Style.RESET_ALL}")
#     report_graph_details(val_graph)

#     print(f"\n{Fore.CYAN}[INFO] Reporting details of test graph...{Style.RESET_ALL}")
#     report_graph_details(test_graph)

#     print(f"\n{Fore.BLUE}[INFO] Shapes of the split data:{Style.RESET_ALL}")
#     print(f"{Fore.BLUE}Training data shape: {train_graph.edge_index.shape}{Style.RESET_ALL}")
#     print(f"{Fore.BLUE}Validation data shape: {val_graph.edge_index.shape}{Style.RESET_ALL}")
#     print(f"{Fore.BLUE}Test data shape: {test_graph.edge_index.shape}{Style.RESET_ALL}")

#     print(f"{Fore.GREEN}✅ Split data into training, validation, and test sets.{Style.RESET_ALL}")
#     print(f"{Fore.GREEN}✅ Negative samples added for link prediction.{Style.RESET_ALL}")

def split_and_save_data(graph_data):
    """
    Split graph data into training/validation/test sets and generate negative samples.
    """
    print(f"{Fore.CYAN}[INFO] Splitting data into training, validation, and test sets...{Style.RESET_ALL}")
    try:
        # Split edges into training, validation, and test sets
        train_edges, val_edges, test_edges = split_edges(graph_data)
        
        # Generate negative samples for each split
        train_neg_samples = add_negative_samples(train_edges, graph_data)
        val_neg_samples = add_negative_samples(val_edges, graph_data)
        test_neg_samples = add_negative_samples(test_edges, graph_data)
        
        # Combine positive and negative samples for each split
        def create_labeled_graph(edges, neg_samples, num_nodes):
            pos_labels = torch.ones(edges.shape[1], dtype=torch.float)  # Positive labels
            neg_labels = torch.zeros(neg_samples.shape[1], dtype=torch.float)  # Negative labels
            
            # Concatenate positive and negative edges
            all_edges = torch.cat([edges, neg_samples], dim=1)
            all_labels = torch.cat([pos_labels, neg_labels])
            
            return torch_geometric.data.Data(
                x=graph_data.x,
                edge_index=all_edges,
                y=all_labels,
                num_nodes=num_nodes
            )
        
        # Create labeled graphs for each split
        train_graph = create_labeled_graph(train_edges, train_neg_samples, graph_data.num_nodes)
        val_graph = create_labeled_graph(val_edges, val_neg_samples, graph_data.num_nodes)
        test_graph = create_labeled_graph(test_edges, test_neg_samples, graph_data.num_nodes)
        
        # Save positive edges and negative samples
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, "graph/positives"), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, "graph/negatives"), exist_ok=True)
        
        torch.save(train_graph, os.path.join(PROCESSED_DATA_DIR, "graph/positives/train_data.pt"))
        torch.save(val_graph, os.path.join(PROCESSED_DATA_DIR, "graph/positives/val_data.pt"))
        torch.save(test_graph, os.path.join(PROCESSED_DATA_DIR, "graph/positives/test_data.pt"))
        
        torch.save(train_neg_samples, os.path.join(PROCESSED_DATA_DIR, "graph/negatives/train_neg_samples.pt"))
        torch.save(val_neg_samples, os.path.join(PROCESSED_DATA_DIR, "graph/negatives/val_neg_samples.pt"))
        torch.save(test_neg_samples, os.path.join(PROCESSED_DATA_DIR, "graph/negatives/test_neg_samples.pt"))
        
        print(f"{Fore.GREEN}✅ Saved split data with labels.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to split or save data: {e}{Style.RESET_ALL}")
        raise


def preprocess():
    """
    Preprocess raw data for a product recommendation system using GNNs.
    Steps include loading/cleaning data, feature engineering, graph construction,
    and splitting data into training/validation/test sets with negative sampling.
    """
    print(f"{Fore.CYAN}[START] Starting data preprocessing pipeline...{Style.RESET_ALL}")

    try:
        cleaned_reviews, cleaned_meta = load_and_clean_data()
        feature_matrix = generate_feature_matrix(cleaned_meta)
        graph_data = build_graph(cleaned_meta, feature_matrix, cleaned_reviews)
        split_and_save_data(graph_data)

        print(f"\n{Fore.GREEN}[SUCCESS] Data preprocessing pipeline completed successfully!{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[FAILURE] Data preprocessing pipeline failed: {e}{Style.RESET_ALL}")
        raise


if __name__ == "__main__":
    preprocess()