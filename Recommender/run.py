import os
import sys
import torch

from logger import get_module_logger

from srcs._0_preprocessing.preprocessor import preprocessor
from srcs._2_training.trainer import trainer
from srcs._2_training.node_embeddings import saving_node_embeddings
from srcs._4_recommending.Recommender import recommender
# =======================
# 📁 Project Setup
# =======================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

logger = get_module_logger("run")


# =======================
# 🚀 Main Pipeline
# =======================
def run():
    logger.info("🔄 Running program started.")

    # # 1) Preprocess
    # logger.info("➡️  Preprocessing data...")
    # preprocessor()
    # logger.info("✅ Preprocessing completed.")

    # # # 2) Train
    # logger.info("➡️  Training model...")
    # trainer()
    # logger.info("✅ Training completed.")

    # # # 3️⃣ Step 3: Save Node Embeddings
    logger.info("➡️  Step 3: Saving node embeddings...")
    saving_node_embeddings()
    logger.info("✅ Step 3 Completed: Node embeddings saved as 'node_embeddings.pt'.")
    
    # # # 4) Recommendation
    # logger.info("➡️  Generating recommendations...")
    # top_recommendations = recommender(user_id=0, top_k=10)
    # logger.info(f"✅ Recommendations completed.")

    # # 5) Display
    # user_id = 0  # or any valid user index
    # top10 = recommendations.get(user_id, [])
    # logger.info(f"Top 10 recommendations for user {user_id}: {top10}")

    logger.info("✅ Program finished successfully.")


# =======================
# 🔧 CLI Entry Point
# =======================
if __name__ == "__main__":
    run()
