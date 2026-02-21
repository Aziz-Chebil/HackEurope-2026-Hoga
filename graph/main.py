"""Main entry point for graph-based Twitter bot detection pipeline.

Usage:
    python main.py train --model gat          # Entraîne le GAT et sauvegarde les poids
    python main.py train --model rgcn         # Entraîne le R-GCN et sauvegarde les poids
    python main.py evaluate --model gat       # Charge les poids et produit résultats + CSV
    python main.py evaluate --model rgcn      # Idem pour R-GCN
"""

import argparse
import os
import random
import time

import numpy as np
import torch

from config import Config
from data_loader import build_graph
from evaluate import compute_probabilities, evaluate_test, save_probabilities
from models import build_model
from train import get_device, train as train_model
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


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Graph-based Twitter bot detection on TwiBot-20"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="Entraîner le modèle")
    train_parser.add_argument("--model", type=str, default="gat", choices=["gat", "rgcn"])
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--hidden_dim", type=int, default=128)
    train_parser.add_argument("--dropout", type=float, default=0.3)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", type=str, default="auto")

    # --- evaluate subcommand ---
    eval_parser = subparsers.add_parser("evaluate", help="Charger un modèle entraîné et produire les résultats")
    eval_parser.add_argument("--model", type=str, default="gat", choices=["gat", "rgcn"])
    eval_parser.add_argument("--hidden_dim", type=int, default=128)
    eval_parser.add_argument("--dropout", type=float, default=0.3)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--device", type=str, default="auto")
    eval_parser.add_argument("--skip_tsne", action="store_true",
                             help="Ne pas générer le t-SNE (peut être lent)")

    args = parser.parse_args()

    cfg = Config()
    cfg.model_type = args.model
    cfg.hidden_dim = args.hidden_dim
    cfg.dropout = args.dropout
    cfg.seed = args.seed
    cfg.device = args.device

    if args.mode == "train":
        cfg.epochs = args.epochs
        cfg.lr = args.lr
        cfg.patience = args.patience

    return cfg, args


def run_train(cfg: Config):
    """Entraîne le modèle et sauvegarde les poids."""
    print("=" * 60)
    print(f"TRAINING — {cfg.model_type.upper()}")
    print("=" * 60)
    print(f"Hidden dim: {cfg.hidden_dim}, Dropout: {cfg.dropout}")
    print(f"LR: {cfg.lr}, Weight decay: {cfg.weight_decay}")
    print(f"Epochs: {cfg.epochs}, Patience: {cfg.patience}")
    print(f"Seed: {cfg.seed}")
    print()

    t0 = time.time()
    data = build_graph(cfg)
    print(f"\nData loading took {time.time() - t0:.1f}s")

    in_dim = data.x.shape[1]
    model = build_model(cfg.model_type, in_dim, cfg)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {cfg.model_type.upper()} ({num_params:,} parameters)")
    print(model)

    t0 = time.time()
    model, history = train_model(model, data, cfg)
    print(f"\nTraining took {time.time() - t0:.1f}s")

    # Plots d'entraînement
    plot_loss_curves(history, cfg)

    checkpoint_path = os.path.join(cfg.checkpoint_dir, f"best_{cfg.model_type}.pt")
    print(f"\nModèle sauvegardé dans : {checkpoint_path}")
    print("Pour produire les résultats : python main.py evaluate "
          f"--model {cfg.model_type}")


def run_evaluate(cfg: Config, args):
    """Charge un modèle entraîné et produit les résultats."""
    checkpoint_path = os.path.join(cfg.checkpoint_dir, f"best_{cfg.model_type}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Erreur : pas de checkpoint trouvé dans {checkpoint_path}")
        print(f"Lance d'abord : python main.py train --model {cfg.model_type}")
        return

    print("=" * 60)
    print(f"EVALUATE — {cfg.model_type.upper()}")
    print("=" * 60)

    data = build_graph(cfg)

    in_dim = data.x.shape[1]
    model = build_model(cfg.model_type, in_dim, cfg)

    device = get_device(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"Poids chargés depuis : {checkpoint_path}")

    model = model.to(device)
    data = data.to(device)

    # Métriques sur le test set
    metrics = evaluate_test(model, data, cfg)

    # Export CSV pour le stacking
    save_probabilities(model, data, cfg)

    # Visualisations
    print("\nGénération des visualisations...")
    plot_confusion_matrix(metrics["confusion_matrix"], cfg)
    plot_probability_distribution(metrics["test_labels"], metrics["test_probs"], cfg)

    if not args.skip_tsne:
        _, indices, embeddings = compute_probabilities(model, data, cfg)
        labels = data.y[indices].cpu().numpy()
        plot_embeddings_tsne(embeddings, labels, cfg)

    print(f"\nDone ! Tous les résultats dans : {cfg.output_dir}")


def main():
    cfg, args = parse_args()
    set_seed(cfg.seed)

    if args.mode == "train":
        run_train(cfg)
    elif args.mode == "evaluate":
        run_evaluate(cfg, args)


if __name__ == "__main__":
    main()
