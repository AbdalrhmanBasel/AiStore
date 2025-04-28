from logger import get_module_logger
from srcs._0_preprocessing_2.graph_construction.load_data import load_and_clean_data
from srcs._0_preprocessing_2.graph_construction.feature_matrix import generate_feature_matrix
from srcs._0_preprocessing_2.graph_construction.graph_builder import graph_builder
from srcs._0_preprocessing_2.graph_construction.EdgeSplitter import split_and_save_data
import os
import sys

logger = get_module_logger("preprocessor")


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)



def preprocessor():
    """
    Main preprocessing pipeline
    """
    try:
        logger.info("==== Starting Preprocessing Pipeline ====")
        
        reviews, meta = load_and_clean_data()
        feature_matrix = generate_feature_matrix(meta)
        # graph = graph_builder(meta, feature_matrix, reviews)
        graph = graph_builder()
        split_and_save_data(graph)
        
        logger.info("==== Preprocessing Completed Successfully ====")
        return True
        
    except Exception as e:
        logger.error("==== Preprocessing Failed ====", exc_info=True)
        return False

