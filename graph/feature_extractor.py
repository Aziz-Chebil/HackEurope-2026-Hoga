"""Extract node features from user profiles."""

import re
from datetime import datetime

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config import Config


def _safe_int(val, default=0) -> int:
    """Parse a value to int, handling strings with spaces and None."""
    if val is None or val == "None" or val == "None ":
        return default
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return default


def _safe_bool(val, default=False) -> bool:
    """Parse a value to bool, handling string representations."""
    if val is None or val == "None" or val == "None ":
        return default
    s = str(val).strip().lower()
    return s in ("true", "1", "yes")


def _safe_str(val, default="") -> str:
    """Return a clean string or default."""
    if val is None or val == "None" or val == "None ":
        return default
    return str(val).strip()


def _parse_created_at(val, collection_date: datetime) -> float:
    """Parse Twitter's created_at and return account age in days."""
    s = _safe_str(val)
    if not s:
        return 0.0
    try:
        dt = datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
        age = (collection_date - dt).total_seconds() / 86400
        return max(age, 0.0)
    except (ValueError, TypeError):
        return 0.0


def extract_features(
    labeled_entries: dict[int, dict],
    num_nodes: int,
    cfg: Config,
) -> torch.Tensor:
    """Extract feature vectors for all nodes.

    Args:
        labeled_entries: dict mapping node index -> entry dict (for labeled users)
        num_nodes: total number of nodes in the graph
        cfg: config object

    Returns:
        Tensor of shape [num_nodes, D] with normalized features.
    """
    collection_date = datetime.strptime(cfg.collection_date, "%Y-%m-%d").replace(
        tzinfo=None
    )
    # We'll parse timezone-aware dates but compare as total seconds

    num_numeric = 14
    num_binary = 6
    D = num_numeric + num_binary

    raw_features = np.zeros((num_nodes, D), dtype=np.float32)

    for idx in range(num_nodes):
        entry = labeled_entries.get(idx)
        if entry is None:
            # Unlabeled neighbor with no profile — features stay zero
            continue

        profile = entry.get("profile", {})

        followers = _safe_int(profile.get("followers_count"))
        friends = _safe_int(profile.get("friends_count"))
        statuses = _safe_int(profile.get("statuses_count"))
        favourites = _safe_int(profile.get("favourites_count"))
        listed = _safe_int(profile.get("listed_count"))

        name = _safe_str(profile.get("name"))
        screen_name = _safe_str(profile.get("screen_name"))
        description = _safe_str(profile.get("description"))
        url = _safe_str(profile.get("url"))
        location = _safe_str(profile.get("location"))

        # Account age
        age_days = _parse_created_at(profile.get("created_at"), collection_date)

        # Numeric features
        row = [
            followers,
            friends,
            statuses,
            favourites,
            listed,
            followers / (friends + 1),
            statuses / (followers + 1),
            listed / (followers + 1),
            age_days,
            statuses / (age_days + 1),  # tweets per day
            len(name),
            len(screen_name),
            len(description),
            sum(c.isdigit() for c in screen_name),
        ]

        # Binary features
        row.extend([
            float(_safe_bool(profile.get("verified"))),
            float(_safe_bool(profile.get("default_profile"))),
            float(_safe_bool(profile.get("default_profile_image"))),
            float(len(url) > 0),
            float(len(location) > 0),
            float(len(description) > 0),
        ])

        raw_features[idx] = row

    # Normalize numeric features with StandardScaler (fit on labeled nodes only)
    labeled_mask = np.array([i in labeled_entries for i in range(num_nodes)])
    scaler = StandardScaler()
    scaler.fit(raw_features[labeled_mask, :num_numeric])
    raw_features[:, :num_numeric] = scaler.transform(raw_features[:, :num_numeric])

    # Replace any NaN/inf from scaling
    raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.tensor(raw_features, dtype=torch.float32)
