"""Main entry point for graph-based Twitter bot detection pipeline."""

import argparse
import random
import sys
import time

import numpy as np
import torch

from config import Config
from data_loader import build_graph
from evaluate import compute_probabilities, evaluate_test, save_probabilities
from models import build_model
from train import train as train_model
from visualize import (
    plot_confusion_matrix,
    plot_embeddings_tsne,
    plot_loss_curves,
    plot_probability_distribution,
)


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> Config:
    """Parse CLI arguments into a Config."""
    parser = argparse.ArgumentParser(
        description="Graph-based Twitter bot detection on TwiBot-20"
    )
    parser.add_argument("--model", type=str, default="gat", choices=["gat", "rgcn"],
                        help="Model architecture (default: gat)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip_tsne", action="store_true",
                        help="Skip t-SNE visualization (can be slow)")

    args = parser.parse_args()
    cfg = Config()
    cfg.model_type = args.model
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.hidden_dim = args.hidden_dim
    cfg.dropout = args.dropout
    cfg.patience = args.patience
    cfg.seed = args.seed
    cfg.device = args.device
    return cfg, args


def main():
    cfg, args = parse_args()
    set_seed(cfg.seed)

    print("=" * 60)
    print("Graph-Based Twitter Bot Detection — TwiBot-20")
    print("=" * 60)
    print(f"Model: {cfg.model_type.upper()}")
    print(f"Hidden dim: {cfg.hidden_dim}, Dropout: {cfg.dropout}")
    print(f"LR: {cfg.lr}, Weight decay: {cfg.weight_decay}")
    print(f"Batch size: {cfg.batch_size}, Neighbors: {cfg.num_neighbors}")
    print(f"Epochs: {cfg.epochs}, Patience: {cfg.patience}")
    print(f"Seed: {cfg.seed}")
    print()

    # Step 1: Load data and build graph
    t0 = time.time()
    data = build_graph(cfg)
    print(f"\nData loading took {time.time() - t0:.1f}s")

    # Step 2: Build model
    in_dim = data.x.shape[1]
    model = build_model(cfg.model_type, in_dim, cfg)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {cfg.model_type.upper()} ({num_params:,} parameters)")
    print(model)

    # Step 3: Train
    t0 = time.time()
    model, history = train_model(model, data, cfg)
    print(f"\nTraining took {time.time() - t0:.1f}s")

    # Step 4: Evaluate on test set
    metrics = evaluate_test(model, data, cfg)

    # Step 5: Save probabilities for stacking
    save_probabilities(model, data, cfg)

    # Step 6: Visualizations
    print("\nGenerating visualizations...")
    plot_loss_curves(history, cfg)
    plot_confusion_matrix(metrics["confusion_matrix"], cfg)
    plot_probability_distribution(metrics["test_labels"], metrics["test_probs"], cfg)

    if not args.skip_tsne:
        # Get embeddings for all labeled nodes for t-SNE
        _, indices, embeddings = compute_probabilities(model, data, cfg)
        labels = data.y[indices].numpy()
        plot_embeddings_tsne(embeddings, labels, cfg)

    print("\nDone! All outputs saved to:", cfg.output_dir)


if __name__ == "__main__":
    main()
