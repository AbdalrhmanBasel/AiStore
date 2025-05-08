import os
import sys
import torch
from torch_geometric.data import Data
from srcs.utils.settings import (
    LEARNING_RATE, EPOCHS, DEVICE,
    MODEL_SAVE_PATH, IMAGES_DIR,
    TRAIN_GRAPH_PATH, VAL_GRAPH_PATH, TEST_GRAPH_PATH
)
from srcs.utils.logger import get_module_logger
from srcs.models.GraphSAGEModel import GraphSAGEModel
from srcs.Recommender import Recommender
from srcs.utils.load_models import load_trained_model

# ----------------------
# Project Setup
# ----------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

logger = get_module_logger("run")

# ----------------------
# Main Pipeline
# ----------------------
def run():
    logger.info("🔄 Starting pipeline")

    # 1) Load pre-split graphs
    logger.info("➡️ Loading train/val/test splits")
    graph_train = torch.load(TRAIN_GRAPH_PATH, weights_only=False)
    graph_val   = torch.load(  VAL_GRAPH_PATH,   weights_only=False)
    graph_test  = torch.load( TEST_GRAPH_PATH,  weights_only=False)
    for name, g in [("train", graph_train), ("val", graph_val), ("test", graph_test)]:
        if not isinstance(g, Data):
            logger.error(f"{name} split is not a PyG Data object")
            return
    logger.info(f"Train: {graph_train.x.shape}, Val: {graph_val.x.shape}, Test: {graph_test.x.shape}")

    # 2) Init model & optimizer
    logger.info("➡️ Initializing GraphSAGE model")
    model     = GraphSAGEModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3) Train (with validation)
    logger.info("➡️ Training with validation")
    train_losses, val_losses = model.train_model(
        train_data=graph_train,
        val_data=graph_val,
        optimizer=optimizer,
        epochs=EPOCHS
    )

    # 4) Plot & save loss curves
    img_path = os.path.join(os.path.dirname(IMAGES_DIR), "training_curve.png")
    GraphSAGEModel.plot_training_curves(train_losses, val_losses, save_path=img_path)

    # 5) Final test evaluation
    logger.info("➡️ Final evaluation on test set")
    # test_loss = model.evaluate_model(graph_test)
    test_loss = model.evaluate_with_metrics(graph_test)
    logger.info(f"✅ Test Loss: {test_loss:.4f}")

    # 6) Save trained model
    logger.info(f"💾 Saving model weights to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

    # 7) Load model for inference
    logger.info("➡️ Loading model for recommendation")
    infer_model = GraphSAGEModel().to(DEVICE)
    infer_model.load(MODEL_SAVE_PATH)

    # 8) Generate and log recommendations
    recommender = Recommender(model_path=MODEL_SAVE_PATH, graph_data=graph_train)
    user_id = 0

    logger.info(f"📦 Top-K Recommendations: {recommender.recommend_top_k_for_user(user_id)}")
    logger.info(f"🎯 Similar Items (for item 0): {recommender.recommend_similar_items(0)}")
    logger.info(f"🧍 Similar Users: {recommender.recommend_similar_users(user_id)}")
    # logger.info(f"🧠 Content-Based: {recommender.recommend_content_based(user_id)}")
    # logger.info(f"🎨 Diverse: {recommender.recommend_diverse_items(user_id)}")
    logger.info(f"🔥 Popular: {recommender.recommend_most_popular()}")
    logger.info(f"🕒 Recent: {recommender.recommend_recent_items() if hasattr(graph_train, 'item_timestamps') else 'N/A'}")
    logger.info(f"🧊 Cold-Start: {recommender.recommend_for_cold_user()}")
    logger.info(f"🧍 Reverse: Recommend users for item 0: {recommender.recommend_users_for_item(0)}")

    logger.info("✅ Pipeline finished successfully.")


if __name__ == "__main__":
    run()
