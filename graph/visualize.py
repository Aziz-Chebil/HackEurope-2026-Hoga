"""Visualization: loss curves, confusion matrix, probability distributions, embeddings."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay

from config import Config


def plot_loss_curves(history: dict, cfg: Config):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{cfg.model_type.upper()} — Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{cfg.model_type.upper()} — Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(cfg.output_dir, f"loss_curves_{cfg.model_type}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curves to {path}")


def plot_confusion_matrix(cm, cfg: Config):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Human", "Bot"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{cfg.model_type.upper()} — Confusion Matrix (Test Set)")
    path = os.path.join(cfg.output_dir, f"confusion_matrix_{cfg.model_type}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {path}")


def plot_probability_distribution(test_labels, test_probs, cfg: Config):
    """Plot overlapping histograms of P(bot) for humans vs bots."""
    fig, ax = plt.subplots(figsize=(8, 5))

    human_probs = test_probs[test_labels == 0]
    bot_probs = test_probs[test_labels == 1]

    ax.hist(human_probs, bins=50, alpha=0.6, label=f"Human (n={len(human_probs)})",
            color="steelblue", density=True)
    ax.hist(bot_probs, bins=50, alpha=0.6, label=f"Bot (n={len(bot_probs)})",
            color="coral", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Threshold (0.5)")
    ax.set_xlabel("P(bot)")
    ax.set_ylabel("Density")
    ax.set_title(f"{cfg.model_type.upper()} — P(bot) Distribution (Test Set)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(cfg.output_dir, f"prob_distribution_{cfg.model_type}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved probability distribution to {path}")


def plot_embeddings_tsne(embeddings, labels, cfg: Config, n_samples: int = 3000):
    """Plot t-SNE of GNN embeddings colored by label."""
    if embeddings is None:
        print("No embeddings available for t-SNE plot.")
        return

    # Filter to labeled nodes only
    mask = labels >= 0
    embeddings = embeddings[mask]
    labels = labels[mask]

    # Subsample if too many
    if len(embeddings) > n_samples:
        rng = np.random.RandomState(cfg.seed)
        idx = rng.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    print(f"Running t-SNE on {len(embeddings)} embeddings...")
    tsne = TSNE(n_components=2, random_state=cfg.seed, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))
    for label_val, name, color in [(0, "Human", "steelblue"), (1, "Bot", "coral")]:
        mask = labels == label_val
        ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.5,
                   label=name, color=color)
    ax.set_title(f"{cfg.model_type.upper()} — t-SNE of GNN Embeddings")
    ax.legend(markerscale=5)
    ax.set_xticks([])
    ax.set_yticks([])

    path = os.path.join(cfg.output_dir, f"tsne_embeddings_{cfg.model_type}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE plot to {path}")
