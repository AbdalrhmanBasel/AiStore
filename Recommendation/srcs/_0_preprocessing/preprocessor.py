import os
import sys
from logger import get_module_logger
from srcs._0_preprocessing.data_loading.load_data import load_reviews_dataset, load_metadata_dataset
from srcs._0_preprocessing.data_cleaning.reviews_dataset_cleaning import clean_review_dataset
from srcs._0_preprocessing.data_cleaning.meta_dataset_clearning import clean_metadata_dataset
from srcs._0_preprocessing.graph_construction.data_mapping import map_users_products_to_pyg
from srcs._0_preprocessing.graph_construction.split_edges import split_edge_index

logger = get_module_logger("preprocessor")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)


def preprocessor():
    """
    Main preprocessing pipeline
    """
    try:
        logger.info("==== Starting Preprocessing Pipeline ====")

        # Load raw datasets
        reviews_df = load_reviews_dataset()
        meta_df = load_metadata_dataset()
        
        # Clean datasets
        cleaned_reviews_df = clean_review_dataset(reviews_df)
        cleaned_meta_df = clean_metadata_dataset(meta_df)

        # Extract user and product IDs
        users_ids = cleaned_reviews_df['user_id']
        products_ids = cleaned_meta_df['parent_asin']

        # Build graph and mappings
        graph, mappings = map_users_products_to_pyg(
            users_ids,
            products_ids,
            metadata=cleaned_meta_df,
            interactions=cleaned_reviews_df
        )

        # Split edges into train/val/test
        train_edge_index, val_edge_index, test_edge_index = split_edge_index(graph.edge_index)

        # Attach splits to graph object for DataLoader
        graph.train_edge_index = train_edge_index
        graph.val_edge_index   = val_edge_index
        graph.test_edge_index  = test_edge_index

        logger.info("==== Preprocessing Completed Successfully ====")
        return graph, mappings, train_edge_index, val_edge_index, test_edge_index

    except Exception as e:
        logger.error("==== Preprocessing Failed ====", exc_info=True)
        return False
