import os
import sys
import json

import pandas as pd
from logger import get_module_logger

from srcs._0_preprocessing.data_loaders.load_reviews_dataset import load_reviews_dataset
from srcs._0_preprocessing.data_loaders.load_meta_dataset import load_metadata_dataset
from srcs._0_preprocessing.data_cleaning.reviews_dataset_cleaning import clean_review_dataset
from srcs._0_preprocessing.data_cleaning.meta_dataset_clearning import clean_metadata_dataset
from srcs._0_preprocessing.utils.save_data import (
    save_as_csv,
    save_as_jsonl,
    save_graph,
    save_graph_splits,
    save_mappings
)
from srcs._0_preprocessing.data_loaders.GraphDataset import GraphDataset

from settings import (
    SAMPLE_CLEANED_REVIEW_PATH,
    SAMPLE_CLEANED_META_PATH,
    GRAPH_DATA_DIR,
    NODE_MAPPING_PATH,
    GRAPH_SPLIT_DIR,
)

logger = get_module_logger("preprocessor")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def preprocessor():
    """
    Main preprocessing pipeline:
      1) Load sampled reviews
      2) Load matching metadata
      3) Clean and save both datasets
      4) Build GraphDataset (with train/val/test split & negative sampling)
      5) Save full graph tensors
      6) Save train/val/test edge splits (pos & neg)
      7) Save node mappings
      8) Return dataset and mappings
    """
    logger.info("==== Starting Preprocessing Pipeline ====")

    # 1. Load & sample reviews
    reviews_df = load_reviews_dataset()
    logger.info(f"✅ Sampled reviews: {len(reviews_df):,} rows")

    # 2. Load only those metadata ASINs
    asins = reviews_df["parent_asin"].unique().tolist()
    meta_df = load_metadata_dataset(asins_to_keep=asins)
    logger.info(f"✅ Loaded metadata for {len(meta_df):,} products")

    # 3. Clean & persist reviews
    cleaned_reviews = clean_review_dataset(reviews_df)
    save_as_csv(cleaned_reviews, SAMPLE_CLEANED_REVIEW_PATH)

    # 4. Clean & persist metadata
    cleaned_meta = clean_metadata_dataset(meta_df)
    save_as_jsonl(cleaned_meta, SAMPLE_CLEANED_META_PATH)

    # 5. Build GraphDataset (splits & negative sampling inside)
    dataset = GraphDataset(
        review_df=cleaned_reviews,
        metadata_df=cleaned_meta,
        val_ratio=0.1,
        test_ratio=0.1,
        negative_sample_ratio=1
    )
    logger.info("✅ GraphDataset constructed (with train/val/test splits & negative samples)")

    # 6. Save full graph tensors
    edge_index, node_feats, labels = dataset.get_graph_data()
    save_graph(edge_index, node_feats, labels, GRAPH_DATA_DIR)
    logger.info(f"✅ Saved full graph to {GRAPH_DATA_DIR}")

    # 7. Save train/val/test edge splits (positive & negative)
    splits = {
        "train_pos": [dataset.edges[i] for i in dataset.train_idx],
        "val_pos":   [dataset.edges[i] for i in dataset.val_idx],
        "test_pos":  [dataset.edges[i] for i in dataset.test_idx],
        
        "train_neg": dataset.neg_edges[:len(dataset.train_idx)],
        "val_neg":   dataset.neg_edges[len(dataset.train_idx):len(dataset.train_idx)+len(dataset.val_idx)],
        "test_neg":  dataset.neg_edges[len(dataset.train_idx)+len(dataset.val_idx):]
    }
    save_graph_splits(splits, GRAPH_SPLIT_DIR)
    logger.info(f"✅ Saved graph splits to {GRAPH_SPLIT_DIR}")

    # 8. Save node‐ID mappings
    mappings = dataset.get_mappings()
    save_mappings(mappings, NODE_MAPPING_PATH)
    logger.info(f"✅ Saved mappings to {NODE_MAPPING_PATH}")

    # 9. Log a few sample node features
    logger.info("✅ Sample node feature vectors (first 5):")
    all_feats = dataset.get_all_node_features()
    for i in range(min(5, all_feats.size(0))):
        logger.info(f"Node {i} features: {all_feats[i].tolist()}")

    logger.info("==== Preprocessing Completed Successfully ====")
    return dataset, mappings
