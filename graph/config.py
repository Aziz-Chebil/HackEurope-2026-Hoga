"""Hyperparameters and paths for graph-based bot detection."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths — data_dir is resolved at runtime (Kaggle download or local)
    data_dir: str = ""
    output_dir: str = os.path.join(os.path.dirname(__file__), "output")
    checkpoint_dir: str = os.path.join(os.path.dirname(__file__), "checkpoints")

    # Model
    model_type: str = "gat"  # "gat" or "rgcn"
    hidden_dim: int = 128
    gat_heads: int = 4
    num_relations: int = 2  # following, follower
    dropout: float = 0.3

    # Training
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 100
    patience: int = 10
    batch_size: int = 512
    num_neighbors: list = field(default_factory=lambda: [15, 10])

    # Misc
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    collection_date: str = "2022-02-01"  # TwiBot-20 collection reference date

    def __post_init__(self):
        self.output_dir = str(Path(self.output_dir).resolve())
        self.checkpoint_dir = str(Path(self.checkpoint_dir).resolve())
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def resolve_data_dir(self):
        """Resolve data_dir: use local path if set, otherwise download from Kaggle."""
        if self.data_dir and Path(self.data_dir).exists():
            self.data_dir = str(Path(self.data_dir).resolve())
            return

        # Try local archive/ folder next to project
        local_archive = Path(__file__).parent.parent / "archive"
        if local_archive.exists() and (local_archive / "train.json").exists():
            self.data_dir = str(local_archive.resolve())
            print(f"Using local data: {self.data_dir}")
            return

        # Download from Kaggle
        from download_data import download_twibot20
        self.data_dir = download_twibot20()
