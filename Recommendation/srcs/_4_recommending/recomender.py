from logger import get_module_logger
from srcs._0_preprocessing.data_loading.dataset import GraphDataset

logger = get_module_logger("recommender")
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def recommender(model, data_loader, device="cpu", top_k=10):
    """
    Generate recommendations based on the model's predictions.

    Args:
        model (torch.nn.Module): The trained recommendation model.
        data_loader (DataLoader): DataLoader that provides the input data.
        device (str, optional): Device to run the recommendation process on. Defaults to "cpu".
        top_k (int, optional): Number of top recommendations to return. Defaults to 10.
    
    Returns:
        dict: A dictionary containing the top-k recommendations for each user.
    """
    logger.info("🔄 Starting recommendation process.")

    model.eval()
    recommendations = {}

    for batch in data_loader:
        batch = batch.to(device)

        scores = model(batch.x, batch.edge_index)

        for user_idx, score in enumerate(scores):
            top_k_indices = score.topk(top_k).indices
            recommendations[user_idx] = top_k_indices.cpu().numpy().tolist()

    logger.info(f"✅ Recommendation process completed.")
    
    return recommendations
