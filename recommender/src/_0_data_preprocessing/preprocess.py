import os
import sys
import torch

from colorama import Fore, Style, init
import torch_geometric

init()

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)

from src._0_data_preprocessing.data_cleaning.clean_reviews import clean_reviews
from src._0_data_preprocessing.data_cleaning.clean_meta import clean_meta
from src._0_data_preprocessing.graph_construction.load_graphs import load_data
from src._0_data_preprocessing.graph_construction.split_graph import split_and_save_data
from src._0_data_preprocessing.graph_construction.graph_builder import construct_graph, save_graph
from src._0_data_preprocessing.graph_construction.report_graph import report_graph_details
from src._0_data_preprocessing.graph_construction.features_matrix import (
    create_feature_matrix, feature_engineering, normalize_features,
)

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