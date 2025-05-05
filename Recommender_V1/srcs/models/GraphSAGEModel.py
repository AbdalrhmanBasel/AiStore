import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from srcs.utils.logger import get_module_logger
from srcs.utils.settings import (
    IN_CHANNELS, HIDDEN_CHANNELS, NUM_LAYERS, DROPOUT, DEVICE,
    LEARNING_RATE, EPOCHS, TRAIN_GRAPH_PATH, VAL_GRAPH_PATH, TEST_GRAPH_PATH,
    MODEL_SAVE_PATH, NUM_USERS
)

# Ensure project root in path
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

logger = get_module_logger("GraphSAGEModel")

class GraphSAGEModel(nn.Module):
    """GraphSAGE-based GNN for link prediction on bipartite user–item graphs."""

    def __init__(
        self,
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        device=DEVICE
    ):
        super().__init__()
        self.device = device
        self.dropout = dropout

        # Build GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.to(self.device)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def train_model(self, train_data, val_data, optimizer, epochs):
        train_data, val_data = train_data.to(self.device), val_data.to(self.device)

        train_losses, val_losses = [], []
        precisions, recalls, f1s = [], [], []

        logger.info(f"🚀 Starting training for {epochs} epochs on {self.device}")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.train()
            optimizer.zero_grad()

            # --- forward & compute training loss using split labels ---
            emb = self(train_data.x, train_data.edge_index)
            edge_idx = train_data.edge_label_index
            labels   = train_data.edge_label.float().to(self.device)
            scores   = (emb[edge_idx[0]] * emb[edge_idx[1]]).sum(dim=1)
            loss_train = F.binary_cross_entropy_with_logits(scores, labels)
            loss_train.backward()
            optimizer.step()

            # --- validation: loss and metrics ---
            loss_val = self.evaluate_model(val_data)
            prec, rec, f1 = self.evaluate_with_metrics(val_data)

            # record histories
            train_losses.append(loss_train.item())
            val_losses.append(loss_val)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            # --- logging ---
            dt = time.time() - t0
            msg = (
                f"[Epoch {epoch}/{epochs}] "
                f"Train Loss: {loss_train:.4f} | Val Loss: {loss_val:.4f} | "
                f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Time: {dt:.2f}s"
            )
            logger.info(msg)

        logger.info("✅ Training complete")
        return train_losses, val_losses

    def evaluate_model(self, data):
        self.eval()
        data = data.to(self.device)
        with torch.no_grad():
            emb = self(data.x, data.edge_index)
            edge_idx = data.edge_label_index
            labels   = data.edge_label.float().to(self.device)
            scores   = (emb[edge_idx[0]] * emb[edge_idx[1]]).sum(dim=1)
            loss     = F.binary_cross_entropy_with_logits(scores, labels)
        return loss.item()

    def evaluate_with_metrics(self, data):
        self.eval()
        data = data.to(self.device)
        with torch.no_grad():
            emb      = self(data.x, data.edge_index)
            edge_idx = data.edge_label_index
            labels   = data.edge_label.long().cpu()
            scores   = (emb[edge_idx[0]] * emb[edge_idx[1]]).sum(dim=1)
            probs    = torch.sigmoid(scores).cpu()
            preds    = (probs > 0.5).long()

        prec = precision_score(labels, preds)
        rec  = recall_score(labels, preds)
        f1   = f1_score(labels, preds)
        # logger.info(f"📊 Metrics — Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        return prec, rec, f1

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"💾 Model saved to {path}")

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        logger.info(f"📥 Model loaded from {path}")

    def get_user_item_embeddings(self, data):
        """Return `(user_embs, item_embs)` slices from the full embedding matrix."""
        self.eval()
        data = data.to(self.device)
        with torch.no_grad():
            emb = self(data.x, data.edge_index)
        return emb[:NUM_USERS], emb[NUM_USERS:]


    @staticmethod
    def plot_training_curves(train_losses, val_losses, save_path="training_curve.png"):
        """Interactive & saved plot of loss curves."""
        epochs = list(range(1, len(train_losses) + 1))
        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses,   label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"📈 Training curve saved to {save_path}")
        plt.show()

def plot_training_metrics(train_losses, val_losses, precisions, recalls, f1s, save_path="training_metrics.png"):
    """Plot and save training and validation curves for loss and metrics."""
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot Training and Validation Loss
    axes[0, 0].plot(train_losses, label="Train Loss", color='blue')
    axes[0, 0].plot(val_losses, label="Val Loss", color='red')
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Plot Precision
    axes[0, 1].plot(precisions, label="Precision", color='green')
    axes[0, 1].set_title("Precision over Epochs")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend()

    # Plot Recall
    axes[1, 0].plot(recalls, label="Recall", color='orange')
    axes[1, 0].set_title("Recall over Epochs")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].legend()

    # Plot F1 Score
    axes[1, 1].plot(f1s, label="F1 Score", color='purple')
    axes[1, 1].set_title("F1 Score over Epochs")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.show()
    logger.info(f"📈 Training metrics saved to {save_path}")