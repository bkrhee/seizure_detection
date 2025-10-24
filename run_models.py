#!/usr/bin/env python3
"""Command-line runner for multiple seizure detection models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from data_utils import SplitConfig, ensure_output_dir, load_dataset, reshape_flattened_signals, stratified_splits
from models.scicnn import SciCNNConfig, compute_metrics as scicnn_metrics, make_loader as scicnn_loader, train_and_evaluate as train_scicnn
from models.snn_model import SNNConfig, compute_metrics as snn_metrics, make_loader as snn_loader, rate_encode, train_and_evaluate as train_snn
from models.svm_classifier import SVMConfig, train_and_evaluate as train_svm
from models.transformer_model import TransformerConfig, compute_metrics as transformer_metrics, train_and_evaluate as train_transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seizure detection models with consistent preprocessing.")
    parser.add_argument("--model", choices=["svm", "scicnn", "snn", "transformer"], required=True, help="Model to train/evaluate.")
    parser.add_argument("--data-path", type=Path, default=Path("EEG_Scaled_data.csv"), help="Path to the dataset CSV.")
    parser.add_argument("--target-column", type=str, default="target", help="Name of the target column in the dataset.")
    parser.add_argument("--channels", type=int, default=None, help="Number of EEG channels (required for CNN/Transformer models).")
    parser.add_argument("--timesteps", type=int, default=None, help="Number of timesteps per channel (required for CNN/Transformer models).")
    parser.add_argument("--test-size", type=float, default=0.1, help="Fraction of data reserved for testing.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction of data reserved for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--input-maps",
        type=int,
        default=8,
        help="Number of per-sensor feature maps (e.g., raw + filtered variants) for SciCNN.",
    )
    parser.add_argument("--svd-components", type=int, default=256, help="Components for SVM TruncatedSVD.")
    parser.add_argument("--svm-C", type=float, default=1.0, help="Inverse regularization strength for SVM.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs for neural models.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory to save models/metrics.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU execution for neural models.")
    return parser.parse_args()


def select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_svm(pipeline, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    scores = pipeline.decision_function(X_test)
    preds = (scores >= 0.0).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, scores)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["average_precision"] = average_precision_score(y_test, scores)
    except ValueError:
        metrics["average_precision"] = float("nan")
    return metrics


def evaluate_neural_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    metric_fn,
    threshold: float,
) -> Dict[str, float]:
    model.eval()
    scores = []
    targets = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            probs = torch.sigmoid(logits)
            scores.append(probs.cpu().numpy())
            targets.append(batch_y.cpu().numpy())
    y_scores = np.concatenate(scores, axis=0)
    y_targets = np.concatenate(targets, axis=0)
    return metric_fn(y_targets, y_scores, threshold)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")

    X_df, y_series = load_dataset(args.data_path, args.target_column)
    config = SplitConfig(test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(X_df, y_series, config)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)

    results = {}

    if args.model == "svm":
        svm_config = SVMConfig(svd_components=args.svd_components, C=args.svm_C)
        pipeline, val_metrics = train_svm(X_train, y_train, X_val, y_val, svm_config)
        test_metrics = evaluate_svm(pipeline, X_test, y_test)
        results = {"val": val_metrics, "test": test_metrics}
        model_artifact = pipeline
    elif args.model == "scicnn":
        if args.channels is None or args.timesteps is None:
            raise ValueError("Channels and timesteps must be specified for SciCNN.")
        X_train_seq = reshape_flattened_signals(X_train_std, args.channels, args.timesteps, feature_maps=args.input_maps)
        X_val_seq = reshape_flattened_signals(X_val_std, args.channels, args.timesteps, feature_maps=args.input_maps)
        X_test_seq = reshape_flattened_signals(X_test_std, args.channels, args.timesteps, feature_maps=args.input_maps)

        device = select_device(args.no_gpu)
        logging.info("Using device: %s", device)
        scicnn_config = SciCNNConfig(
            sensors=args.channels,
            samples=args.timesteps,
            input_maps=args.input_maps,
            epochs=args.epochs,
        )
        model, val_metrics = train_scicnn(X_train_seq, y_train, X_val_seq, y_val, scicnn_config, device)
        X_test_scicnn = np.expand_dims(X_test_seq, axis=1) if X_test_seq.ndim == 3 else X_test_seq
        test_loader = scicnn_loader(X_test_scicnn, y_test, scicnn_config.batch_size, shuffle=False)
        model_artifact = model
        results = {
            "val": val_metrics,
            "test": evaluate_neural_model(model, test_loader, device, scicnn_metrics, scicnn_config.threshold),
        }
    elif args.model == "snn":
        device = select_device(args.no_gpu)
        logging.info("Using device: %s", device)
        snn_config = SNNConfig(
            channels=args.channels or X_train.shape[1],
            timesteps=args.timesteps or 1,
            epochs=args.epochs,
        )
        model, val_metrics = train_snn(X_train_std, y_train, X_val_std, y_val, snn_config, device)
        model_artifact = model

        test_loader = snn_loader(X_test_std, y_test, snn_config.batch_size, shuffle=False)

        model.eval()
        scores = []
        targets = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                encoded = rate_encode(batch_x, snn_config.spike_timesteps).to(device)
                _, mem_rec = model(encoded)
                logits = mem_rec[-1].squeeze(-1)
                probs = torch.sigmoid(logits)
                scores.append(probs.cpu().numpy())
                targets.append(batch_y.cpu().numpy())
        results = {
            "val": val_metrics,
            "test": snn_metrics(np.concatenate(targets, axis=0), np.concatenate(scores, axis=0), snn_config.threshold),
        }
    elif args.model == "transformer":
        if args.channels is None or args.timesteps is None:
            raise ValueError("Channels and timesteps must be specified for the Transformer.")
        X_train_seq = reshape_flattened_signals(X_train_std, args.channels, args.timesteps)
        X_val_seq = reshape_flattened_signals(X_val_std, args.channels, args.timesteps)
        X_test_seq = reshape_flattened_signals(X_test_std, args.channels, args.timesteps)

        device = select_device(args.no_gpu)
        logging.info("Using device: %s", device)
        transformer_config = TransformerConfig(
            channels=args.channels,
            timesteps=args.timesteps,
            epochs=args.epochs,
        )
        model, val_metrics = train_transformer(X_train_seq, y_train, X_val_seq, y_val, transformer_config, device)
        model_artifact = model
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_test_seq.astype(np.float32)),
                torch.from_numpy(y_test.astype(np.float32)),
            ),
            batch_size=transformer_config.batch_size,
            shuffle=False,
        )
        results = {
            "val": val_metrics,
            "test": evaluate_neural_model(model, test_loader, device, transformer_metrics, transformer_config.threshold),
        }
    else:
        raise ValueError(f"Unsupported model '{args.model}'.")

    logging.info("Validation metrics: %s", json.dumps(results["val"], indent=2))
    logging.info("Test metrics: %s", json.dumps(results["test"], indent=2))

    if args.output_dir:
        output_dir = ensure_output_dir(args.output_dir)
        metrics_path = output_dir / f"{args.model}_metrics.json"
        with metrics_path.open("w") as fp:
            json.dump(results, fp, indent=2)
        if args.model == "svm":
            import joblib

            joblib.dump(model_artifact, output_dir / "svm_pipeline.joblib")
        else:
            torch.save(model_artifact.state_dict(), output_dir / f"{args.model}_weights.pt")
        logging.info("Artifacts saved to %s", output_dir)


if __name__ == "__main__":
    main()
