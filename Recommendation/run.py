# from srcs._0_preprocessing.graph_construction.graph_builder import graph_builder
from srcs._0_preprocessing.preprocessor import preprocessor
from srcs._2_training.utils.hyperparameters_tuner import hyperparameters_tuner
from srcs._2_training.trainer import trainer
from srcs._3_evaluating.evaluate import evaluater
from srcs._4_recommending.recomender import recommender
from logger import get_module_logger

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

logger = get_module_logger("run")

def run():
    logger.info("🔄 Running program started.")
    # review_df, meta_df =load_data()
    graph, mappings, train_edge_index, val_edge_index, test_edge_index = preprocessor()
    # hyperparameters_tuner()
    trainer(graph, mappings, train_edge_index, val_edge_index, test_edge_index)
    # evaluater()
    # recommender()
    logger.info("✅ program running completed.")


if __name__ == "__main__":
    run()


