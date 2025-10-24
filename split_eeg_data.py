#!/usr/bin/env python3
"""Split the EEG dataset into stratified train/validation CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratified train/validation split for EEG dataset.")
    parser.add_argument("--data-path", type=Path, default=Path("EEG_Scaled_data.csv"), help="Path to the full CSV dataset.")
    parser.add_argument("--target-column", type=str, default="target", help="Name of the target column in the CSV file.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Fraction of samples to allocate to the validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train-output", type=Path, default=Path("train_split.csv"), help="Output CSV path for the training split.")
    parser.add_argument("--val-output", type=Path, default=Path("val_split.csv"), help="Output CSV path for the validation split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data_path)
    if args.target_column not in df.columns:
        raise KeyError(f"Target column '{args.target_column}' not found in dataset.")

    y = df[args.target_column]

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y,
    )

    train_df.to_csv(args.train_output, index=False)
    val_df.to_csv(args.val_output, index=False)

    print(f"Saved training split to {args.train_output} ({len(train_df)} rows).")
    print(f"Saved validation split to {args.val_output} ({len(val_df)} rows).")


if __name__ == "__main__":
    main()
