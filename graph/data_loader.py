"""Load TwiBot-20 dataset and construct PyG Data object."""

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from config import Config
from feature_extractor import extract_features


def load_split(filepath: str) -> list[dict]:
    """Load a JSON split file."""
    print(f"  Loading {Path(filepath).name}...")
    with open(filepath) as f:
        return json.load(f)


def build_graph(cfg: Config) -> Data:
    """Build a PyG Data object from TwiBot-20 JSON files.

    Returns a Data object with:
        - x: node features [N, D]
        - edge_index: [2, E]
        - edge_type: [E] (0=following, 1=follower)
        - y: labels [N] (0=human, 1=bot, -1=unlabeled)
        - train_mask, val_mask, test_mask: boolean masks [N]
    """
    # Ensure data is available (download from Kaggle if needed)
    cfg.resolve_data_dir()

    print("Loading dataset splits...")
    train_data = load_split(str(Path(cfg.data_dir) / "train.json"))
    val_data = load_split(str(Path(cfg.data_dir) / "dev.json"))
    test_data = load_split(str(Path(cfg.data_dir) / "test.json"))

    # Collect all labeled users with their split info
    all_labeled = []
    for entry in train_data:
        entry["_split"] = "train"
        all_labeled.append(entry)
    for entry in val_data:
        entry["_split"] = "val"
        all_labeled.append(entry)
    for entry in test_data:
        entry["_split"] = "test"
        all_labeled.append(entry)

    print(f"  Labeled users: {len(all_labeled)} "
          f"(train={len(train_data)}, val={len(val_data)}, test={len(test_data)})")

    # Build user_id -> index mapping; labeled users first, then unlabeled neighbors
    user_id_to_idx = {}
    labeled_entries_by_idx = {}

    for entry in all_labeled:
        uid = str(entry["ID"])
        if uid not in user_id_to_idx:
            idx = len(user_id_to_idx)
            user_id_to_idx[uid] = idx
            labeled_entries_by_idx[idx] = entry

    num_labeled = len(user_id_to_idx)

    # Discover unlabeled neighbor IDs
    for entry in all_labeled:
        if not entry.get("neighbor"):
            continue
        for rel in ("following", "follower"):
            for nid in entry["neighbor"].get(rel, []):
                nid = str(nid)
                if nid not in user_id_to_idx:
                    user_id_to_idx[nid] = len(user_id_to_idx)

    num_nodes = len(user_id_to_idx)
    num_unlabeled = num_nodes - num_labeled
    print(f"  Total nodes: {num_nodes} ({num_labeled} labeled, {num_unlabeled} unlabeled neighbors)")

    # Build edges
    print("Building edge index...")
    src_list, dst_list, etype_list = [], [], []

    for entry in all_labeled:
        uid = str(entry["ID"])
        src_idx = user_id_to_idx[uid]
        if not entry.get("neighbor"):
            continue
        # following: user -> target (edge type 0)
        for nid in entry["neighbor"].get("following", []):
            dst_idx = user_id_to_idx[str(nid)]
            src_list.append(src_idx)
            dst_list.append(dst_idx)
            etype_list.append(0)
        # follower: target -> user (edge type 1)
        for nid in entry["neighbor"].get("follower", []):
            follower_idx = user_id_to_idx[str(nid)]
            src_list.append(follower_idx)
            dst_list.append(src_idx)
            etype_list.append(1)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(etype_list, dtype=torch.long)
    print(f"  Edges: {edge_index.shape[1]} (following={etype_list.count(0)}, follower={etype_list.count(1)})")

    # Extract node features
    print("Extracting node features...")
    x = extract_features(labeled_entries_by_idx, num_nodes, cfg)
    print(f"  Feature matrix: {x.shape}")

    # Build labels and masks
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for idx, entry in labeled_entries_by_idx.items():
        y[idx] = int(entry["label"])
        split = entry["_split"]
        if split == "train":
            train_mask[idx] = True
        elif split == "val":
            val_mask[idx] = True
        elif split == "test":
            test_mask[idx] = True

    print(f"  Labels: {(y == 0).sum().item()} human, {(y == 1).sum().item()} bot, "
          f"{(y == -1).sum().item()} unlabeled")
    print(f"  Masks: train={train_mask.sum().item()}, val={val_mask.sum().item()}, "
          f"test={test_mask.sum().item()}")

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.num_nodes = num_nodes
    data.user_ids = list(user_id_to_idx.keys())

    return data
