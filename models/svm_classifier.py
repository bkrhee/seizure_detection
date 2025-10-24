"""Support Vector Machine baseline for seizure detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass
class SVMConfig:
    variance_threshold: float = 0.0
    svd_components: int = 256
    C: float = 1.0
    max_iter: int = 5000
    tol: float = 1e-4


def build_pipeline(config: SVMConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("variance", VarianceThreshold(config.variance_threshold)),
            ("scale_in", StandardScaler(with_mean=False)),
            ("svd", TruncatedSVD(n_components=config.svd_components, algorithm="randomized", random_state=42)),
            ("scale_out", StandardScaler()),
            ("svm", LinearSVC(C=config.C, class_weight="balanced", max_iter=config.max_iter, tol=config.tol)),
        ]
    )


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: SVMConfig,
) -> Tuple[Pipeline, Dict[str, float]]:
    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    decision_scores = pipeline.decision_function(X_val)
    predictions = (decision_scores >= 0.0).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions, zero_division=0),
        "recall": recall_score(y_val, predictions, zero_division=0),
        "f1": f1_score(y_val, predictions, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_val, decision_scores)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["average_precision"] = average_precision_score(y_val, decision_scores)
    except ValueError:
        metrics["average_precision"] = float("nan")
    return pipeline, metrics
