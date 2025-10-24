"""Utility helpers for EEG seizure prediction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitConfig:
    test_size: float = 0.1
    val_size: float = 0.1
    seed: int = 42


def load_dataset(csv_path: Path, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not present in {csv_path}.")
    y = df[target_column].astype("int64")
    X = df.drop(columns=[target_column])
    return X, y


def stratified_splits(
    X: pd.DataFrame,
    y: pd.Series,
    config: SplitConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_size = config.test_size
    val_fraction_within_train = config.val_size / max(1.0 - test_size, 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config.seed,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_fraction_within_train,
        random_state=config.seed,
        stratify=y_train,
    )
    return (
        X_train.to_numpy(dtype=np.float32),
        X_val.to_numpy(dtype=np.float32),
        X_test.to_numpy(dtype=np.float32),
        y_train.to_numpy(dtype=np.int64),
        y_val.to_numpy(dtype=np.int64),
        y_test.to_numpy(dtype=np.int64),
    )


def apply_standard_scaler(
    train_features: np.ndarray,
    other_features: Tuple[np.ndarray, ...],
) -> Tuple[np.ndarray, ...]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    transformed = [train_scaled]
    for block in other_features:
        transformed.append(scaler.transform(block))
    return tuple(transformed)


def reshape_flattened_signals(
    X: np.ndarray,
    channels: int,
    timesteps: int,
    feature_maps: int = 1,
) -> np.ndarray:
    expected = channels * timesteps * feature_maps
    if X.shape[1] != expected:
        raise ValueError(
            f"Cannot reshape array with {X.shape[1]} features into shape ({feature_maps}, {channels}, {timesteps})."
        )
    if feature_maps == 1:
        return X.reshape(X.shape[0], channels, timesteps)
    return X.reshape(X.shape[0], feature_maps, channels, timesteps)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
