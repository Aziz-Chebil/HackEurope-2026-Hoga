"""Inference function: predict bot probabilities for specific dataset rows."""

import os

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from data_loader import build_graph
from models import build_model
from train import get_device


def predict(rows: list[dict], model_type: str = "rgcn") -> np.ndarray:
    """Compute P(bot) for the given dataset rows using a trained graph model.

    Since GNN predictions depend on the full graph structure (neighbor
    message-passing), the entire graph is loaded and inference is run on
    all nodes.  Only the probabilities for the requested rows are returned.

    Args:
        rows: List of dataset entries (dicts with an ``"ID"`` field),
            typically a subset of the TwiBot-20 JSON data.
        model_type: ``"rgcn"`` (default) or ``"gat"``.

    Returns:
        numpy array of shape ``(len(rows), 2)`` where column 0 is
        P(human) and column 1 is P(bot) for each requested user.

    Raises:
        ValueError: If *model_type* is not ``"rgcn"`` or ``"gat"``.
        FileNotFoundError: If no checkpoint exists for the requested model.
        KeyError: If a user ID from *rows* is not present in the dataset graph.
    """
    if model_type not in ("rgcn", "gat"):
        raise ValueError(f"model_type must be 'rgcn' or 'gat', got '{model_type}'")

    user_ids = [str(entry["ID"]) for entry in rows]

    # --- config ---------------------------------------------------------------
    cfg = Config()
    cfg.model_type = model_type

    checkpoint_path = os.path.join(cfg.checkpoint_dir, f"best_{model_type}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            f"Train the model first: python main.py train --model {model_type}"
        )

    # --- build graph & model --------------------------------------------------
    data = build_graph(cfg)

    uid_to_idx = {uid: idx for idx, uid in enumerate(data.user_ids)}

    # Validate requested user IDs
    missing = [uid for uid in user_ids if uid not in uid_to_idx]
    if missing:
        raise KeyError(
            f"{len(missing)} user ID(s) not found in the graph: {missing[:10]}"
        )

    indices = [uid_to_idx[uid] for uid in user_ids]

    in_dim = data.x.shape[1]
    model = build_model(model_type, in_dim, cfg)

    device = get_device(cfg)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    data = data.to(device)
    model.eval()

    # --- inference ------------------------------------------------------------
    with torch.no_grad():
        edge_type = data.edge_type if model_type == "rgcn" else None
        logits = model(data.x, data.edge_index, edge_type)
        probs = F.softmax(logits, dim=1)  # (N, 2)

    # Select only the requested users
    result = probs[indices].cpu().numpy()  # (len(user_ids), 2)

    return result
