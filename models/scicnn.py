"""SciCNN architecture with Neural Pattern Clustering head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SciCNNConfig:
    sensors: int
    samples: int
    input_maps: int = 8  # 1 raw trace + 7 band-limited maps by default
    primary_kernels: Tuple[Tuple[int, int], ...] = ((1, 16), (1, 8), (1, 4))
    secondary_kernels: Tuple[Tuple[int, int], ...] = ((1, 8), (1, 4), (1, 2))
    pool_sizes: Tuple[Tuple[int, int], ...] = ((1, 4), (1, 4), (1, 8))
    branch_channels: Tuple[int, ...] = (8, 16, 32)
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 30
    patience: int = 5
    threshold: float = 0.5
    seed: int = 42
    npc_dim: int = 64
    npc_centroids: int = 256
    npc_weight: float = 1e-3
    centroid_temperature: float = 1.0


class SciCNN(nn.Module):
    def __init__(self, config: SciCNNConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList()
        in_channels = config.input_maps

        for idx, out_channels in enumerate(config.branch_channels):
            primary_kernel = config.primary_kernels[idx]
            secondary_kernel = config.secondary_kernels[idx]
            pool_size = config.pool_sizes[idx]

            block = ParallelBlock(
                in_channels=in_channels,
                primary_out=out_channels,
                secondary_out=out_channels,
                primary_kernel=primary_kernel,
                secondary_kernel=secondary_kernel,
                pool_size=pool_size,
                dropout=config.dropout,
            )
            self.blocks.append(block)
            in_channels = out_channels * 2  # concatenated branches

        with torch.no_grad():
            dummy = torch.zeros(1, config.input_maps, config.sensors, config.samples)
            for block in self.blocks:
                dummy = block(dummy)
            flattened_dim = int(np.prod(dummy.shape[1:]))

        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, config.npc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.npc_layer = NeuralPatternClustering(
            feature_dim=config.npc_dim,
            num_centroids=config.npc_centroids,
            temperature=config.centroid_temperature,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Expect shape (batch, maps, sensors, samples)
        h = x
        for block in self.blocks:
            h = block(h)
        features = self.feature_head(h)
        logits, penalty = self.npc_layer(features)
        return logits.squeeze(-1), penalty


class ParallelBlock(nn.Module):
    """Two-branch convolutional block mirroring SciCNN design."""

    def __init__(
        self,
        in_channels: int,
        primary_out: int,
        secondary_out: int,
        primary_kernel: Tuple[int, int],
        secondary_kernel: Tuple[int, int],
        pool_size: Tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.primary_branch = nn.Sequential(
            nn.Conv2d(in_channels, primary_out, kernel_size=primary_kernel, padding=(0, primary_kernel[1] // 2)),
            nn.BatchNorm2d(primary_out),
            nn.ReLU(inplace=True),
        )
        self.secondary_branch = nn.Sequential(
            nn.Conv2d(in_channels, secondary_out, kernel_size=secondary_kernel, padding=(0, secondary_kernel[1] // 2)),
            nn.BatchNorm2d(secondary_out),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary_branch(x)
        secondary = self.secondary_branch(x)
        concat = torch.cat([primary, secondary], dim=1)
        pooled = self.pool(concat)
        return self.dropout(pooled)


class NeuralPatternClustering(nn.Module):
    """NPC head that maintains learnable centroids in feature space."""

    def __init__(self, feature_dim: int, num_centroids: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.centroids = nn.Parameter(torch.randn(num_centroids, feature_dim))
        self.temperature = temperature
        self.classifier = nn.Linear(num_centroids, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # features: (batch, feature_dim)
        # Compute Euclidean distances to centroids
        distances = torch.cdist(features.unsqueeze(1), self.centroids.unsqueeze(0))  # (batch, 1, num_centroids)
        distances = distances.squeeze(1)
        min_distance, _ = torch.min(distances, dim=1)

        # Soft assignment based on distance
        scaled = torch.softmax(-distances / max(self.temperature, 1e-6), dim=1)
        logits = self.classifier(scaled)

        # Use mean of minimal distances as NPC penalty (encourages centroids to follow data)
        penalty = min_distance.mean()
        return logits, penalty


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features.astype(np.float32)), torch.from_numpy(labels.astype(np.float32)))
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
    config: SciCNNConfig,
    device: torch.device,
    ) -> Tuple[SciCNN, Dict[str, float]]:
    torch.manual_seed(config.seed)
    model = SciCNN(config).to(device)

    X_train_prepared = _ensure_4d(X_train, config)
    X_val_prepared = _ensure_4d(X_val, config)

    train_loader = make_loader(X_train_prepared, y_train, config.batch_size, shuffle=True)
    val_loader = make_loader(X_val_prepared, y_val, config.batch_size, shuffle=False)

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
    metrics: Dict[str, float] = {}

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        samples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, npc_penalty = model(batch_x)
            loss = criterion(logits, batch_y)
            loss = loss + config.npc_weight * npc_penalty
            loss.backward()
            optimizer.step()

            batch_size = batch_x.size(0)
            train_loss += float(loss.item()) * batch_size
            samples += batch_size

        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total = 0
            scores = []
            targets = []
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                logits, npc_penalty = model(val_x)
                loss = criterion(logits, val_y)
                loss = loss + config.npc_weight * npc_penalty
                probs = torch.sigmoid(logits)
                scores.append(probs.cpu().numpy())
                targets.append(val_y.cpu().numpy())
                total_loss += float(loss.item()) * val_x.size(0)
                total += val_x.size(0)
            avg_val_loss = total_loss / max(total, 1)
            y_scores = np.concatenate(scores, axis=0)
            y_targets = np.concatenate(targets, axis=0)
            metrics = compute_metrics(y_targets, y_scores, config.threshold)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, metrics


def _ensure_4d(X: np.ndarray, config: SciCNNConfig) -> np.ndarray:
    if X.ndim == 4:
        expected = (config.input_maps, config.sensors, config.samples)
        if tuple(X.shape[1:]) != expected:
            raise ValueError(f"SciCNN expected features shaped (*, {expected}), got {X.shape}.")
        return X
    if X.ndim == 3:
        if X.shape[1] != config.sensors or X.shape[2] != config.samples:
            raise ValueError(
                f"SciCNN expected features with sensors={config.sensors} and samples={config.samples}, got {X.shape}."
            )
        return np.expand_dims(X, axis=1)
    raise ValueError(f"SciCNN expects 3D or 4D arrays, got shape {X.shape}.")
