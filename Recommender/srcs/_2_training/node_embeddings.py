import os
import sys
import torch

from logger import get_module_logger
from settings import EMBEDDINGS_SAVE_PATH, TRAINED_MODEL_PATH

from srcs._2_training.utils.load_model import load_model
from srcs._2_training.utils.load_model import load_graph_data
from srcs._1_modeling.GraphSAGE import GraphSAGE

logger = get_module_logger("node_embeddings.py")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def saving_node_embeddings():
    """
    Load the model, graph data, and save node embeddings.
    """
    logger.info("➡️  Step 3: Saving node embeddings...")

    # Load model and graph data
    model = GraphSAGE()  # Instantiate the GraphSAGE model
    model, _, _ = load_model(model=model, checkpoint_path=TRAINED_MODEL_PATH)  # Load the model from checkpoint
    graph = load_graph_data()  # Load the graph data

    # Ensure model has a 'get_embeddings' method
    if hasattr(model, 'get_embeddings'):
        # Get node embeddings using the model
        logger.info("➡️  Generating node embeddings...")
        node_embeddings = model.get_embeddings(graph.x, graph.edge_index)
        
        # Save node embeddings to a file
        torch.save(node_embeddings, EMBEDDINGS_SAVE_PATH)
        logger.info(f"✅ Step 3 Completed: Node embeddings saved to '{EMBEDDINGS_SAVE_PATH}'.")
    else:
        logger.error("❌ Model does not have 'get_embeddings' method. Please verify your model.")
