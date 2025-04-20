# Create a visualization script that includes the key visualizations:
# 1. Training/Validation Loss Plot
# 2. ROC Curve
# 3. Degree Distribution
# 4. Top-K Recommendation Scores

# We'll use dummy functions/data where required as placeholders

import os
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc
import networkx as nx

VISUALIZATION_DIR = "./visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(VISUALIZATION_DIR, "loss_curve.png"))
    plt.close()

def plot_roc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(VISUALIZATION_DIR, "roc_curve.png"))
    plt.close()

def plot_degree_distribution(edge_index):
    src_nodes = edge_index[0]
    degrees = torch.bincount(src_nodes)
    plt.figure(figsize=(7, 5))
    plt.hist(degrees.tolist(), bins=50)
    plt.yscale('log')
    plt.xlabel("Node Degree")
    plt.ylabel("Count (log scale)")
    plt.title("Degree Distribution")
    plt.grid(True)
    plt.savefig(os.path.join(VISUALIZATION_DIR, "degree_distribution.png"))
    plt.close()

def plot_top_k_recommendations(scores, user_id, top_k=10):
    top_scores, top_indices = torch.topk(scores, top_k)
    plt.figure(figsize=(8, 5))
    plt.bar(range(top_k), top_scores.tolist())
    plt.xticks(range(top_k), [f"Item {i.item()}" for i in top_indices], rotation=45)
    plt.title(f"Top-{top_k} Recommendations for User {user_id}")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, f"user_{user_id}_top_k.png"))
    plt.close()

