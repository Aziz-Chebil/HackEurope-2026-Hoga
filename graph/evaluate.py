"""Evaluation metrics and probability generation for the stacking ensemble."""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import Config
from train import get_device


@torch.no_grad()
def compute_probabilities(model, data, cfg: Config):
    """Compute P(bot) for all labeled nodes (full-batch).

    Returns:
        probs: numpy array with P(bot) for each labeled node
        indices: node indices corresponding to each probability
        embeddings: numpy array of embeddings for each labeled node
    """
    device = get_device(cfg)
    model = model.to(device)
    data = data.to(device)
    model.eval()

    use_edge_type = cfg.model_type == "rgcn"
    edge_type = data.edge_type if use_edge_type else None
    logits = model(data.x, data.edge_index, edge_type)

    # Get all labeled node indices
    labeled_mask = data.y >= 0
    labeled_indices = labeled_mask.nonzero(as_tuple=True)[0].cpu().numpy()

    probs = F.softmax(logits[labeled_mask], dim=1)[:, 1].cpu().numpy()

    embeddings = None
    if hasattr(model, "embedding"):
        embeddings = model.embedding[labeled_mask].cpu().numpy()

    return probs, labeled_indices, embeddings


def evaluate_test(model, data, cfg: Config) -> dict:
    """Evaluate model on the test set and print metrics."""
    probs, indices, _ = compute_probabilities(model, data, cfg)

    # Build split info for each labeled node
    labels = data.y[indices].cpu().numpy()
    test_filter = data.test_mask[indices].cpu().numpy()

    test_labels = labels[test_filter]
    test_probs = probs[test_filter]
    test_preds = (test_probs >= 0.5).astype(int)

    acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)
    prec = precision_score(test_labels, test_preds)
    rec = recall_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)

    print("\n" + "=" * 50)
    print(f"TEST SET RESULTS ({cfg.model_type.upper()})")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=['Human', 'Bot'])}")

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
        "test_labels": test_labels,
        "test_probs": test_probs,
    }


def save_probabilities(model, data, cfg: Config) -> str:
    """Save P(bot) for all labeled users to CSV for the stacking ensemble.

    Output CSV columns: user_id, split, true_label, p_bot_graph
    """
    probs, indices, _ = compute_probabilities(model, data, cfg)

    rows = []
    for i, idx in enumerate(indices):
        idx = int(idx)
        uid = data.user_ids[idx]
        label = int(data.y[idx].item())
        if data.train_mask[idx]:
            split = "train"
        elif data.val_mask[idx]:
            split = "val"
        else:
            split = "test"
        rows.append({
            "user_id": uid,
            "split": split,
            "true_label": label,
            "p_bot_graph": float(probs[i]),
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(cfg.output_dir, f"graph_probabilities_{cfg.model_type}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved probabilities to {out_path}")
    print(f"  Total: {len(df)}, Train: {(df.split=='train').sum()}, "
          f"Val: {(df.split=='val').sum()}, Test: {(df.split=='test').sum()}")
    return out_path
