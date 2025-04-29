import os
import sys
from logger import get_module_logger
from srcs._0_preprocessing.preprocessor import preprocessor
from srcs._2_training.trainer import trainer
from srcs._2_training.data_loaders import create_data_loaders
from srcs._4_recommending.recomender import recommender

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

logger = get_module_logger("run")

def run():
    logger.info("🔄 Running program started.")

    # 1) Preprocess
    logger.info("➡️  Preprocessing data...")
    graph, mappings, train_eidx, val_eidx, test_eidx = preprocessor()
    logger.info("✅ Preprocessing completed.")

    # 2) Train
    logger.info("➡️  Training model...")
    model = trainer(graph, mappings, train_eidx, val_eidx, test_eidx)
    logger.info("✅ Training completed.")

    # 3) Recommendation
    logger.info("➡️  Generating recommendations...")
    _, _, test_loader = create_data_loaders(graph)
    recommendations = recommender(model, test_loader, device="cpu", top_k=5)
    logger.info(f"✅ Recommendations completed.")

    # 4) Display
    user_id = 0  # or any valid user index
    top5 = recommendations.get(user_id, [])
    logger.info(f"Top 5 recommendations for user {user_id}: {top5}")

    logger.info("✅ Program finished successfully.")

if __name__ == "__main__":
    run()
