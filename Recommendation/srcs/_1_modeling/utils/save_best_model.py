import os
import sys
import torch
from logger import get_module_logger

logger = get_module_logger("best_model_saver")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
sys.path.append(PROJECT_ROOT)


class BestModelSaver:
    def __init__(self, mode="max", save_dir="checkpoints/best", model_name="best_model"):
        """
        Args:
            mode (str): "max" if higher metric is better, "min" if lower is better.
            save_dir (str): Directory where best model will be saved.
            model_name (str): Base filename for saved model and embeddings.
        """
        assert mode in ["max", "min"], "Mode must be 'max' or 'min'"
        self.best_score = None
        self.mode = mode
        self.save_dir = save_dir
        self.model_name = model_name

        os.makedirs(save_dir, exist_ok=True)

    def is_improved(self, score):
        if self.best_score is None:
            return True
        return (score > self.best_score) if self.mode == "max" else (score < self.best_score)

    def save_if_best(self, model, user_embeddings, item_embeddings, current_score):
        """
        Save model if current_score is better than best_score.

        Args:
            model (torch.nn.Module): Trained model.
            user_embeddings (torch.Tensor): User embeddings.
            item_embeddings (torch.Tensor): Item embeddings.
            current_score (float): Metric to compare.
        """
        if self.is_improved(current_score):
            self.best_score = current_score

            model_path = os.path.join(self.save_dir, f"{self.model_name}_model.pth")
            embeddings_path = os.path.join(self.save_dir, f"{self.model_name}_embeddings.pt")

            torch.save(model.state_dict(), model_path)
            torch.save({
                'user_embeddings': user_embeddings,
                'item_embeddings': item_embeddings
            }, embeddings_path)

            logger.info(f"[BestModelSaver] Improved score: {current_score:.4f}. Model saved to {model_path}")
