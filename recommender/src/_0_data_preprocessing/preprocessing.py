import os
import torch
import pandas as pd


from utils.data import load_data
from utils.clean_reviews import clean_reviews
from utils.clean_meta import clean_meta
from utils.features_matrix import (
    create_feature_matrix,
    feature_engineering,
    normalize_features
)
from utils.graphs import construct_graph, save_graph
from utils.split_data import split_edges, create_data_for_link_prediction




def preprocess_data():
    # === Define All Paths ===
    raw_reviews_path = "../../data/raw/reviews_electronics_small.csv"
    raw_meta_path = "../../data/raw/metadata_electronics_small.jsonl"

    samples_dir = "../../data/raw/samples/"
    processed_dir = "../../data/processed/"
    graph_save_path = "../../data/processed/graph.pt"
    feature_matrix_path = os.path.join(processed_dir, "meta_feature_matrix.csv")

    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    sample_reviews_path = os.path.join(samples_dir, "sample_reviews.csv")
    sample_meta_path = os.path.join(samples_dir, "sample_metadata.json")
    cleaned_reviews_path = os.path.join(samples_dir, "cleaned_reviews.csv")
    cleaned_meta_path = os.path.join(processed_dir, "cleaned_meta.csv")

    # === Step 1: Load and Clean Raw Data ===
    print("🔄 Loading raw reviews and metadata...")
    reviews_df, meta_df = load_data(raw_reviews_path, raw_meta_path)

    print("🧹 Cleaning datasets...")
    cleaned_reviews = clean_reviews(reviews_df)
    cleaned_meta = clean_meta(meta_df)

    # === Step 2: Save Raw Samples and Cleaned Outputs ===
    print("💾 Saving dataset samples and cleaned versions...")
    reviews_df.to_csv(sample_reviews_path, index=False)
    meta_df.to_json(sample_meta_path, orient="records", lines=True)
    cleaned_reviews.to_csv(cleaned_reviews_path, index=False)
    cleaned_meta.to_csv(cleaned_meta_path, index=False)

    print(f"✅ Saved: {sample_reviews_path}, {sample_meta_path}")
    print(f"✅ Cleaned datasets saved to: {cleaned_reviews_path}, {cleaned_meta_path}")
    print(f"📊 Raw Shapes — Reviews: {reviews_df.shape}, Meta: {meta_df.shape}")
    print(f"📊 Cleaned Shapes — Reviews: {cleaned_reviews.shape}, Meta: {cleaned_meta.shape}")

    # === Step 3: Feature Engineering ===
    print("\n⚙️ Generating feature matrix...")
    feature_matrix = create_feature_matrix(cleaned_meta)
    feature_matrix = feature_engineering(feature_matrix)
    feature_matrix = normalize_features(feature_matrix)
    feature_matrix.to_csv(feature_matrix_path, index=False)
    print(f"✅ Feature matrix saved to: {feature_matrix_path}")

    # === Step 4: Preview Final Tables ===
    print("\n🔍 Preview — Cleaned Reviews:")
    print(cleaned_reviews.head(5))
    print("\n🔍 Preview — Cleaned Metadata:")
    print(cleaned_meta.head(5))
    print("\n🔍 Preview — Feature Matrix:")
    print(feature_matrix.head(5))

    # === Step 5: Graph Construction ===
    print("\n🔗 Building product-user interaction graph...")

    node_features = [
        "price",
        "average_rating",
        "rating_number",
        "num_features",
        "description_length",
        "price_per_rating",
        "feature_to_rating_ratio"
    ]

    # Ensure all required features are present in the matrix
    missing_features = [f for f in node_features if f not in feature_matrix.columns]
    if missing_features:
        raise ValueError(f"🚨 Missing required features in feature matrix: {missing_features}")

    graph_data = construct_graph(
        meta_df=cleaned_meta,
        meta_features_df=feature_matrix,
        reviews_df=reviews_df,
        product_col='parent_asin',
        user_col='user_id',
        label_col=None  # or set if you're doing classification
    )


    save_graph(graph_data, graph_save_path)
    print(f"✅ Graph saved to: {graph_save_path}")


    # === Step 6: Split data ===
    print("\n🔀 Splitting data into training, validation, and test sets...")

    # Load the full Data object (disable weights_only mode)
    graph_data = torch.load(graph_save_path, map_location='cpu', weights_only=False)

    # Split edges into training, validation, and test sets
    train_data, val_data, test_data = split_edges(graph_data)

    # Prepare data for link prediction by adding negative samples
    # We use the graph data to generate negative samples for training, validation, and test data
    def add_negative_samples(data, num_samples=10000):
        # This function should implement the logic to add negative samples for link prediction
        # For now, this is a placeholder. You can use a negative sampling approach from `torch_geometric`.
        negative_samples = torch.randint(0, graph_data.num_nodes, (2, num_samples))  # Example of random negative edges
        return negative_samples

    train_neg_samples = add_negative_samples(train_data)
    val_neg_samples = add_negative_samples(val_data)
    test_neg_samples = add_negative_samples(test_data)

    # Save the split data to files for later use
    train_data_path = os.path.join(processed_dir, "train_data.pt")
    val_data_path = os.path.join(processed_dir, "val_data.pt")
    test_data_path = os.path.join(processed_dir, "test_data.pt")

    torch.save(train_data, train_data_path)
    torch.save(val_data, val_data_path)
    torch.save(test_data, test_data_path)

    # Save negative samples to files
    train_neg_samples_path = os.path.join(processed_dir, "train_neg_samples.pt")
    val_neg_samples_path = os.path.join(processed_dir, "val_neg_samples.pt")
    test_neg_samples_path = os.path.join(processed_dir, "test_neg_samples.pt")

    torch.save(train_neg_samples, train_neg_samples_path)
    torch.save(val_neg_samples, val_neg_samples_path)
    torch.save(test_neg_samples, test_neg_samples_path)

    print(f"✅ Saved split data: {train_data_path}, {val_data_path}, {test_data_path}")
    print(f"✅ Saved negative samples data for training, validation, and testing.")

    # Report the shapes of the datasets
    print(f"\n📊 Shapes of the split data:")
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    print(f"✅ Split data into training, validation, and test sets.")
    print(f"✅ Negative samples added for link prediction.")

# Run the pipeline
if __name__ == "__main__":
    preprocess_data()
