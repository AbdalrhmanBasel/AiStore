import os
import json
import torch
import pandas as pd
from logger import get_module_logger

logger = get_module_logger("save_data")


def save_as_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame as CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"💾 Saved {len(df):,} records to {path} as CSV")


def save_as_jsonl(df: pd.DataFrame, path: str, orient: str = "records", lines: bool = True, force_ascii: bool = False) -> None:
    """
    Save DataFrame as JSON Lines (JSONL).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient=orient, lines=lines, force_ascii=force_ascii)
    logger.info(f"💾 Saved {len(df):,} records to {path} as JSONL")


def save_as_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame as Parquet.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"💾 Saved {len(df):,} records to {path} as Parquet")


def save_mappings(mappings: dict, path: str) -> None:
    """
    Save node-user and node-product mappings as JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2)
    logger.info(f"💾 Saved mappings to {path}")


def save_graph(edges: torch.Tensor, node_features: torch.Tensor, labels: torch.Tensor, dir_path: str) -> None:
    """
    Save graph data as .pt files in a directory.
    """
    os.makedirs(dir_path, exist_ok=True)
    torch.save(edges, os.path.join(dir_path, "edge_index.pt"))
    torch.save(node_features, os.path.join(dir_path, "features.pt"))
    torch.save(labels, os.path.join(dir_path, "labels.pt"))
    logger.info(f"💾 Saved graph tensors to {dir_path}")


def save_graph_splits(splits: dict, dir_path: str) -> None:
    """
    Save train/val/test edge splits (both positive & negative) as torch tensors.
    Expects splits keys: train_pos, val_pos, test_pos, train_neg, val_neg, test_neg,
    each mapping to a list of [src, dst] pairs.
    """
    os.makedirs(dir_path, exist_ok=True)

    # Mapping the split names to the aliases used in the trainer
    alias_map = {
        "train_pos": "train_data.pt",
        "val_pos":   "val_data.pt",
        "test_pos":  "test_data.pt",
        "train_neg": "train_neg.pt",
        "val_neg":   "val_neg.pt",
        "test_neg":  "test_neg.pt",
    }

    for split_name, edge_list in splits.items():
        # Convert list of [u, v] pairs into a 2xN LongTensor
        if len(edge_list) == 0:
            tensor = torch.empty((2, 0), dtype=torch.long)
        else:
            arr = torch.tensor(edge_list, dtype=torch.long)  # shape (N,2)
            tensor = arr.t().contiguous()                   # shape (2,N)

        # Get the alias for the split (if available)
        filename = alias_map.get(split_name, f"{split_name}.pt")
        path = os.path.join(dir_path, filename)

        # Save the tensor
        torch.save(tensor, path)
        logger.info(f"💾 Saved {split_name} edges ({tensor.size(1)} samples) to {path}")