"""Spiking neural network baseline using snnTorch surrogate gradients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SNNConfig:
    channels: int
    timesteps: int
    hidden_dim: int = 512
    spike_timesteps: int = 20
    beta: float = 0.95
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 20
    patience: int = 5
    threshold: float = 0.5
    seed: int = 42


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(features.astype(np.float32)), torch.from_numpy(labels.astype(np.float32))),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def rate_encode(batch: torch.Tensor, timesteps: int) -> torch.Tensor:
    scaled = torch.sigmoid(batch)  # map to (0,1)
    expanded = scaled.unsqueeze(0).expand(timesteps, -1, -1)
    spikes = torch.bernoulli(expanded)
    return spikes


def build_model(input_dim: int, config: SNNConfig) -> nn.Module:
    try:
        import snntorch as snn  # type: ignore
        from snntorch import surrogate
    except ImportError as exc:  # pragma: no cover
        raise ImportError("snnTorch is required for the spiking model. Install with `pip install snntorch`.") from exc

    spike_grad = surrogate.fast_sigmoid()

    class SNNNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, config.hidden_dim)
            self.lif1 = snn.Leaky(beta=config.beta, spike_grad=spike_grad, init_hidden=True)
            self.fc2 = nn.Linear(config.hidden_dim, 1)
            self.lif2 = snn.Leaky(beta=config.beta, spike_grad=spike_grad, init_hidden=True, output=True)

        def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_size = inputs.size(1)
            mem1 = self.lif1.init_leaky(batch_size=batch_size, device=inputs.device)
            mem2 = self.lif2.init_leaky(batch_size=batch_size, device=inputs.device)

            spk2_rec = []
            mem2_rec = []
            for step in range(inputs.size(0)):
                cur1 = self.fc1(inputs[step])
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_rec.append(spk2)
                mem2_rec.append(mem2)
            return torch.stack(spk2_rec), torch.stack(mem2_rec)

    return SNNNet()


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
    config: SNNConfig,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, float]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    input_dim = X_train.shape[1]
    model = build_model(input_dim, config).to(device)

    train_loader = make_loader(X_train, y_train, config.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, config.batch_size, shuffle=False)

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
        total_loss = 0.0
        total_samples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            encoded = rate_encode(batch_x, config.spike_timesteps).to(device)
            optimizer.zero_grad(set_to_none=True)
            spk_rec, mem_rec = model(encoded)
            # Use final membrane potential as logit estimate
            logits = mem_rec[-1].squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = batch_x.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        avg_train_loss = total_loss / max(total_samples, 1)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            count = 0
            scores = []
            targets = []
            for val_x, val_y in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                encoded = rate_encode(val_x, config.spike_timesteps).to(device)
                spk_rec, mem_rec = model(encoded)
                logits = mem_rec[-1].squeeze(-1)
                loss = criterion(logits, val_y)
                probs = torch.sigmoid(logits)
                scores.append(probs.cpu().numpy())
                targets.append(val_y.cpu().numpy())
                val_loss += float(loss.item()) * val_x.size(0)
                count += val_x.size(0)
            avg_val_loss = val_loss / max(count, 1)
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
