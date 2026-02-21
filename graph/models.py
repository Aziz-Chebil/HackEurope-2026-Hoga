"""GAT and R-GCN models for bot detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, RGCNConv


class GATBotDetector(nn.Module):
    """Graph Attention Network for bot detection.

    Architecture:
        GATConv(D, 128, heads=4) -> ELU -> Dropout
        -> GATConv(512, 128, heads=1) -> ELU
        -> Linear(128, 2)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1,
                             concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type=None):
        # edge_type is ignored for GAT
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        self.embedding = x  # save for visualization
        logits = self.classifier(x)
        return logits


class RGCNBotDetector(nn.Module):
    """Relational Graph Convolutional Network for bot detection.

    Architecture:
        RGCNConv(D, 128, num_relations=2) -> ReLU -> Dropout
        -> RGCNConv(128, 128, num_relations=2) -> ReLU
        -> Linear(128, 2)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128,
                 num_relations: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        self.embedding = x  # save for visualization
        logits = self.classifier(x)
        return logits


def build_model(model_type: str, in_dim: int, cfg) -> nn.Module:
    """Factory to create the requested model."""
    if model_type == "gat":
        return GATBotDetector(
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            heads=cfg.gat_heads,
            dropout=cfg.dropout,
        )
    elif model_type == "rgcn":
        return RGCNBotDetector(
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            num_relations=cfg.num_relations,
            dropout=cfg.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
