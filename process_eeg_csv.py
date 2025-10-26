#!/usr/bin/env python3
"""Utility CLI to post-process the windowed EEG CSV into train/val/test splits.

The CSV produced by `preprocess_chbmit.py` includes four metadata columns
(`subject`, `record`, `start_sec`, `end_sec`), a binary `target`, and a dense set
of flattened feature columns (`f00000`, `f00001`, ...). This script:

* separates metadata from numerical features to avoid leaking identifiers;
* performs stratified train/validation/test splits with configurable sizes;
* writes the feature arrays to a compressed `.npz` archive and the metadata for
  each split to companion CSV files; and
* records a JSON summary describing the resulting dataset.

Downstream training scripts can load `splits/eeg_dataset_splits.npz` and the
metadata CSV files to align samples with their original recordings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import ensure_output_dir


DEFAULT_METADATA_COLUMNS = ("subject", "record", "start_sec", "end_sec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split EEG_Scaled_data.csv into train/val/test arrays.")
    parser.add_argument("--input-csv", type=Path, default=Path("EEG_Scaled_data.csv"), help="Path to the preprocessed CSV.")
    parser.add_argument("--target-column", type=str, default="target", help="Name of the binary target column.")
    parser.add_argument(
        "--metadata-columns",
        nargs="*",
        default=None,
        help="Metadata columns to keep separate from features (defaults to subject/record/start_sec/end_sec if present).",
    )
    parser.add_argument("--test-size", type=float, default=0.1, help="Fraction reserved for the test split.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction reserved for the validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified splitting.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits"),
        help="Directory where split arrays/metadata are saved.",
    )
    return parser.parse_args()


def resolve_metadata_columns(df: pd.DataFrame, requested: Sequence[str] | None) -> List[str]:
    if requested:
        missing = [col for col in requested if col not in df.columns]
        if missing:
            joined = ", ".join(missing)
            raise KeyError(f"Metadata columns not found in dataset: {joined}.")
        return list(requested)
    available = [col for col in DEFAULT_METADATA_COLUMNS if col in df.columns]
    return available


def main() -> None:
    args = parse_args()
    if args.test_size < 0 or args.val_size < 0 or args.test_size + args.val_size >= 1:
        raise ValueError("Ensure test_size and val_size are non-negative and sum to less than 1.")

    df = pd.read_csv(args.input_csv)
    if args.target_column not in df.columns:
        raise KeyError(f"Target column '{args.target_column}' not found in {args.input_csv}.")

    metadata_cols = resolve_metadata_columns(df, args.metadata_columns)
    feature_cols = [col for col in df.columns if col not in metadata_cols and col != args.target_column]
    if not feature_cols:
        raise RuntimeError("No feature columns remain after removing metadata/target.")

    metadata_df = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)
    y_series = df[args.target_column].astype("int64")
    feature_df = df[feature_cols].astype("float32")

    # First split off the test set.
    stratify = y_series if y_series.nunique() > 1 else None
    X_train_df, X_test_df, y_train, y_test, meta_train_df, meta_test_df = train_test_split(
        feature_df,
        y_series,
        metadata_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    # Determine validation fraction within the remaining training data.
    val_fraction = args.val_size / max(1.0 - args.test_size, 1e-8)
    stratify_val = y_train if y_train.nunique() > 1 else None
    X_train_df, X_val_df, y_train, y_val, meta_train_df, meta_val_df = train_test_split(
        X_train_df,
        y_train,
        meta_train_df,
        test_size=val_fraction,
        random_state=args.seed,
        stratify=stratify_val,
    )

    output_dir = ensure_output_dir(args.output_dir)

    splits_path = output_dir / "eeg_dataset_splits.npz"
    np.savez_compressed(
        splits_path,
        X_train=X_train_df.to_numpy(dtype=np.float32),
        X_val=X_val_df.to_numpy(dtype=np.float32),
        X_test=X_test_df.to_numpy(dtype=np.float32),
        y_train=y_train.to_numpy(dtype=np.int64),
        y_val=y_val.to_numpy(dtype=np.int64),
        y_test=y_test.to_numpy(dtype=np.int64),
        feature_columns=np.array(feature_cols, dtype=object),
    )

    if metadata_cols:
        meta_train_df.to_csv(output_dir / "metadata_train.csv", index=False)
        meta_val_df.to_csv(output_dir / "metadata_val.csv", index=False)
        meta_test_df.to_csv(output_dir / "metadata_test.csv", index=False)

    summary = {
        "input_csv": str(args.input_csv),
        "feature_columns": feature_cols,
        "metadata_columns": metadata_cols,
        "target_column": args.target_column,
        "num_train": int(len(X_train_df)),
        "num_val": int(len(X_val_df)),
        "num_test": int(len(X_test_df)),
        "class_balance": {
            "train_positive": int(y_train.sum()),
            "train_negative": int(len(y_train) - int(y_train.sum())),
            "val_positive": int(y_val.sum()),
            "val_negative": int(len(y_val) - int(y_val.sum())),
            "test_positive": int(y_test.sum()),
            "test_negative": int(len(y_test) - int(y_test.sum())),
        },
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
    }
    summary_path = output_dir / "split_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[INFO] Saved feature splits to {splits_path}.")
    if metadata_cols:
        print("[INFO] Saved split metadata CSV files alongside the NPZ archive.")
    print(f"[INFO] Summary written to {summary_path}.")


if __name__ == "__main__":
    main()

