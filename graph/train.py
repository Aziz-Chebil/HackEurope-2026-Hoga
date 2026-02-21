"""Training loop — full-batch GNN training on the complete graph.

The graph is small enough (~191K nodes, ~208K edges) to fit in memory,
so we use full-batch forward passes over all nodes.
"""

import os
import time

import torch
import torch.nn.functional as F

from config import Config


def get_device(cfg: Config) -> torch.device:
    """Resolve the device to use."""
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg.device)


def train_epoch(model, data, optimizer, device, use_edge_type: bool, mask):
    """Run one training epoch (full-batch)."""
    model.train()
    optimizer.zero_grad()

    edge_type = data.edge_type if use_edge_type else None
    logits = model(data.x, data.edge_index, edge_type)

    # Loss only on labeled nodes in the given mask
    loss = F.cross_entropy(logits[mask], data.y[mask])
    loss.backward()
    optimizer.step()

    preds = logits[mask].argmax(dim=1)
    acc = (preds == data.y[mask]).float().mean().item()
    return loss.item(), acc


@torch.no_grad()
def eval_epoch(model, data, device, use_edge_type: bool, mask):
    """Evaluate the model on a given mask (full-batch)."""
    model.eval()

    edge_type = data.edge_type if use_edge_type else None
    logits = model(data.x, data.edge_index, edge_type)

    loss = F.cross_entropy(logits[mask], data.y[mask])
    preds = logits[mask].argmax(dim=1)
    acc = (preds == data.y[mask]).float().mean().item()
    return loss.item(), acc


def train(model, data, cfg: Config):
    """Full training loop with early stopping.

    Returns:
        model: trained model
        history: dict with train_loss, val_loss, train_acc, val_acc lists
    """
    device = get_device(cfg)
    print(f"Using device: {device}")
    model = model.to(device)
    data = data.to(device)

    use_edge_type = cfg.model_type == "rgcn"

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_path = os.path.join(cfg.checkpoint_dir, f"best_{cfg.model_type}.pt")

    print(f"\nTraining {cfg.model_type.upper()} for up to {cfg.epochs} epochs "
          f"(patience={cfg.patience})...\n")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, data, optimizer, device, use_edge_type, data.train_mask)
        val_loss, val_acc = eval_epoch(
            model, data, device, use_edge_type, data.val_mask)

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val loss: {best_val_loss:.4f})")
                break

    # Load best model
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    print(f"Loaded best model from {best_path}")

    return model, history
