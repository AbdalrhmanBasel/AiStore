import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_score, recall_score, f1_score

# Ensure project root in path
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from srcs.utils.settings import (
    IN_CHANNELS, HIDDEN_CHANNELS, NUM_LAYERS, DROPOUT, NUM_USERS,
    NUM_NEGATIVE_SAMPLES, DEVICE, BATCH_SIZE
)
from srcs.utils.logger import get_module_logger

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
        self.num_layers = num_layers

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
        """
        Train loop that logs training & validation loss each epoch,
        and returns the loss histories.
        """
        train_data, val_data = train_data.to(self.device), val_data.to(self.device)
        train_losses, val_losses = [], []

        logger.info(f"🚀 Starting training for {epochs} epochs on {self.device}")
        logger.info(f"🔢 Total params: {sum(p.numel() for p in self.parameters())}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            optimizer.zero_grad()

            # --- forward & train loss ---
            emb = self(train_data.x, train_data.edge_index)
            neg_train = self._negative_sampling(train_data)
            loss_train = self._link_pred_loss(emb, train_data.edge_label_index, neg_train)
            loss_train.backward()
            optimizer.step()

            # --- validation ---
            loss_val = self.evaluate_model(val_data)

            # record
            train_losses.append(loss_train.item())
            val_losses.append(loss_val)

            # log
            dt = time.time() - t0
            lr = optimizer.param_groups[0]['lr']
            msg = (
                f"[Epoch {epoch}/{epochs}] "
                f"Train: {loss_train:.4f} | Val: {loss_val:.4f} | "
                f"Time: {dt:.2f}s | LR: {lr:.6f}"
            )
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated(self.device) / 1024**2
                msg += f" | GPU Mem: {mem:.1f}MB"
            logger.info(msg)

        logger.info("✅ Training complete")
        return train_losses, val_losses

    def evaluate_model(self, data):
        """Compute link-prediction loss on `data` (no grad)."""
        self.eval()
        data = data.to(self.device)
        with torch.no_grad():
            emb = self(data.x, data.edge_index)
            neg = self._negative_sampling(data)
            loss = self._link_pred_loss(emb, data.edge_label_index, neg)
        return loss.item()

    def evaluate_with_metrics(self, data):
        """Compute precision/recall/F1 on positive vs negative edges."""
        self.eval()
        data = data.to(self.device)
        with torch.no_grad():
            emb = self(data.x, data.edge_index)
            neg = self._negative_sampling(data)

            pos = (emb[data.edge_label_index[0]] * emb[data.edge_label_index[1]]).sum(1)
            neg = (emb[neg[0]] * emb[neg[1]]).sum(1)
            scores = torch.cat([pos, neg]).cpu()
            preds = (scores > 0).long()
            labels = torch.cat([
                torch.ones_like(pos), torch.zeros_like(neg)
            ]).long().cpu()

        prec = precision_score(labels, preds)
        rec  = recall_score(labels, preds)
        f1   = f1_score(labels, preds)
        logger.info(f"📊 Metrics — P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")
        return prec, rec, f1

    def get_user_item_embeddings(self, data):
        """Return `(user_embs, item_embs)` slices from the full embedding matrix."""
        self.eval()
        data = data.to(self.device)
        with torch.no_grad():
            emb = self(data.x, data.edge_index)
        return emb[:NUM_USERS], emb[NUM_USERS:]

    def recommend_top_k(self, user_embs, item_embs, top_k=10, batch_size=BATCH_SIZE):
        """Batch‐wise top‐K for all users, avoids OOM."""
        idx_list, score_list = [], []
        item_t = item_embs.t()
        for i in range(0, user_embs.size(0), batch_size):
            u = user_embs[i:i+batch_size]
            sc = u @ item_t
            s, idx = sc.topk(top_k, dim=1)
            idx_list.append(idx)
            score_list.append(s)
        return torch.cat(idx_list), torch.cat(score_list)

    def _negative_sampling(self, data):
        e = data.edge_index
        existing = set(zip(e[0].tolist(), e[1].tolist()))
        neg = set()
        N, M = data.num_nodes, NUM_NEGATIVE_SAMPLES
        while len(neg) < M:
            u = torch.randint(0, N, ()).item()
            v = torch.randint(0, N, ()).item()
            if (u, v) not in existing:
                neg.add((u, v))
        neg = torch.tensor(list(neg), dtype=torch.long).t().contiguous()
        return neg.to(self.device)

    def _link_pred_loss(self, emb, pos_idx, neg_idx):
        pos = (emb[pos_idx[0]] * emb[pos_idx[1]]).sum(1)
        neg = (emb[neg_idx[0]] * emb[neg_idx[1]]).sum(1)
        scores = torch.cat([pos, neg])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]).to(self.device)
        return F.binary_cross_entropy_with_logits(scores, labels)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"💾 Model saved to {path}")

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        logger.info(f"📥 Model loaded from {path}")

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
