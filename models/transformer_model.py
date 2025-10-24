"""Transformer-based classifier for seizure detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TransformerConfig:
    channels: int
    timesteps: int
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 30
    patience: int = 5
    threshold: float = 0.5
    seed: int = 42


class ChannelPositionalEncoding(nn.Module):
    def __init__(self, channels: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(channels, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding.unsqueeze(0)


class TransformerClassifier(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.timesteps, config.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.pos_encoding = ChannelPositionalEncoding(config.channels, config.embed_dim)
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(config.embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, timesteps)
        tokens = self.input_proj(x)  # (batch, channels, embed_dim)
        tokens = self.pos_encoding(tokens)
        encoded = self.encoder(tokens)
        encoded = self.layer_norm(encoded)
        pooled = self.pool(encoded.transpose(1, 2)).squeeze(-1)
        logits = self.fc(pooled)
        return logits.squeeze(-1)


def _make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    preds = (y_scores >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["average_precision"] = average_precision_score(y_true, y_scores)
    except ValueError:
        metrics["average_precision"] = float("nan")
    return metrics


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TransformerConfig,
    device: torch.device,
) -> Tuple[TransformerClassifier, Dict[str, float]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = TransformerClassifier(config).to(device)
    train_loader = _make_loader(X_train, y_train, config.batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, config.batch_size, shuffle=False)

    positives = float(np.count_nonzero(y_train == 1))
    negatives = float(len(y_train) - positives)
    if positives == 0.0:
        raise ValueError("Training data contains no positive samples.")
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = batch_x.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        avg_train_loss = running_loss / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            total = 0
            scores = []
            targets = []
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                logits = model(val_x)
                loss = criterion(logits, val_y)
                probs = torch.sigmoid(logits)
                scores.append(probs.cpu().numpy())
                targets.append(val_y.cpu().numpy())
                val_loss += float(loss.item()) * val_x.size(0)
                total += val_x.size(0)
            avg_val_loss = val_loss / max(total, 1)
            y_scores = np.concatenate(scores, axis=0)
            y_targets = np.concatenate(targets, axis=0)
            metrics = compute_metrics(y_targets, y_scores, config.threshold)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, metrics
