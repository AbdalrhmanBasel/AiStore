from srcs._0_preprocessing.graph_builder import graph_builder
from srcs._0_preprocessing.preprocessor import preprocessor
from srcs._2_training.hyperparameters_tuner import hyperparameters_tuner
from srcs._2_training.trainer import trainer
from srcs._3_evaluating.evaluate import evaluater
from srcs._4_recommending.recomender import recommender

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def run():
    graph_builder()
    preprocessor()
    hyperparameters_tuner()
    trainer()
    evaluater()
    recommender()


if __name__ == "__main__":
    run()