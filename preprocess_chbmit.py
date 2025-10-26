#!/usr/bin/env python3
"""Preprocess the CHB-MIT EEG dataset into windowed features suitable for the repo models.

current project conventions:

* read .edf waveforms with MNE (one recording at a time);
* keep a consistent set of EEG channels across all files;
* optionally resample every recording to a shared target sampling rate;
* generate raw plus seven band-limited feature maps (delta through high-gamma);
* slice overlapping windows, label them using the official seizure annotations, and
  flatten to a tabular representation;
* scale the features with a global StandardScaler and persist the preprocessing spec.

The resulting CSV matches what the training scripts expect (`target` column followed
by flattened features). Metadata columns are prepended so downstream analysis can
still group by subject/record if needed.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import mne
    from mne.io import read_raw_edf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing optional dependency 'mne'. Install it first with `pip install mne`."
    ) from exc


# Default spectral bands closely follow the Kaggle reference (raw + seven filters).
# Frequencies are expressed in Hz.
DEFAULT_BANDS: Tuple[Tuple[str, Optional[float], Optional[float]], ...] = (
    ("raw", None, None),
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("beta", 12.0, 18.0),
    ("beta_high", 18.0, 30.0),
    ("gamma_low", 30.0, 45.0),
    ("gamma_high", 45.0, 70.0),
)


@dataclass(frozen=True)
class Annotation:
    """Seizure annotation converted to sample indices."""

    onset: int
    offset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Window and scale CHB-MIT EEG recordings.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to the unpacked CHB-MIT dataset.")
    parser.add_argument(
        "--records-file",
        type=Path,
        default=None,
        help="Path to the RECORDS file (defaults to <data-root>/RECORDS).",
    )
    parser.add_argument(
        "--records-with-seizures",
        type=Path,
        default=None,
        help="Path to the RECORDS-WITH-SEIZURES file (defaults to <data-root>/RECORDS-WITH-SEIZURES).",
    )
    parser.add_argument(
        "--channel-list",
        type=Path,
        default=None,
        help="Optional text file with one channel per line to force a specific channel ordering.",
    )
    parser.add_argument(
        "--target-sfreq",
        type=float,
        default=128.0,
        help="Resample every recording to this sampling frequency (Hz). Set <= 0 to keep native rate.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Window length in seconds for feature slices.",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=0.5,
        help="Stride in seconds between consecutive windows.",
    )
    parser.add_argument(
        "--notch-freq",
        type=float,
        default=60.0,
        help="Apply a single-frequency notch filter at this frequency (Hz). Set to 0 to disable.",
    )
    parser.add_argument(
        "--lowpass",
        type=float,
        default=None,
        help="Optional low-pass cutoff applied before band decomposition.",
    )
    parser.add_argument(
        "--highpass",
        type=float,
        default=0.5,
        help="Optional high-pass cutoff applied before band decomposition.",
    )
    parser.add_argument(
        "--balance-ratio",
        type=float,
        default=1.0,
        help="Maximum number of negative windows kept per positive window. Set <= 0 to keep all.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("EEG_Scaled_data.csv"),
        help="Destination CSV path for the scaled dataset.",
    )
    parser.add_argument(
        "--preprocessor-output",
        type=Path,
        default=Path("outputs/preprocessor.joblib"),
        help="Where to persist the fitted StandardScaler and metadata.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=Path("outputs/preprocessing_summary.json"),
        help="Where to write a JSON summary of the preprocessing run.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of EDF files processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Random seed for negative window subsampling.",
    )
    return parser.parse_args()


def read_records_file(path: Path) -> List[Path]:
    lines = []
    with path.open("r") as handle:
        for line in handle:
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                lines.append(Path(cleaned))
    return lines


def resolve_records(
    data_root: Path,
    records_file: Optional[Path],
    seizure_records_file: Optional[Path],
) -> Tuple[List[Path], List[Path]]:
    records_path = records_file if records_file is not None else data_root / "RECORDS"
    seizure_path = seizure_records_file if seizure_records_file is not None else data_root / "RECORDS-WITH-SEIZURES"

    if not records_path.exists():
        raise FileNotFoundError(f"Missing RECORDS file: {records_path}")

    all_records = read_records_file(records_path)
    seizure_records = read_records_file(seizure_path) if seizure_path.exists() else []

    all_resolved = [(data_root / rel_path).resolve() for rel_path in all_records]
    seizure_resolved = {(data_root / rel_path).resolve() for rel_path in seizure_records}

    missing_files = [path for path in all_resolved if not path.exists()]
    if missing_files:
        missing_str = "\n  ".join(str(p) for p in missing_files)
        raise FileNotFoundError(f"The following EDF files listed in {records_path} were not found:\n  {missing_str}")

    return all_resolved, sorted(seizure_resolved)


def load_channel_template(edf_paths: Sequence[Path], explicit_list: Optional[Path] = None) -> List[str]:
    if explicit_list is not None:
        with explicit_list.open("r") as handle:
            return [line.strip() for line in handle if line.strip()]

    common_channels: Optional[set[str]] = None
    ordered_reference: List[str] = []

    for idx, edf_path in enumerate(edf_paths):
        raw = read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
        channel_types = raw.get_channel_types()
        eeg_channels = [name for name, kind in zip(raw.ch_names, channel_types) if kind == "eeg"]

        if common_channels is None:
            common_channels = set(eeg_channels)
            ordered_reference = eeg_channels
        else:
            common_channels &= set(eeg_channels)
        raw.close()

    if not common_channels:
        raise RuntimeError("Could not determine a common set of EEG channels across the dataset.")

    template = [ch for ch in ordered_reference if ch in common_channels]
    if not template:
        raise RuntimeError("Common channel intersection is empty after aligning with the reference ordering.")
    return template


def parse_seizure_annotation(annotation_path: Path, sfreq: float) -> List[Annotation]:
    if not annotation_path.exists():
        return []

    starts: List[float] = []
    ends: List[float] = []
    pattern = re.compile(r"(\d+(?:\.\d+)?)")

    with annotation_path.open("r") as handle:
        for line in handle:
            lowered = line.strip().lower()
            if "start" in lowered:
                match = pattern.search(line)
                if match:
                    starts.append(float(match.group(1)))
            elif "end" in lowered:
                match = pattern.search(line)
                if match:
                    ends.append(float(match.group(1)))

    annotations = []
    for onset, offset in zip(starts, ends):
        start_sample = int(round(onset * sfreq))
        end_sample = int(round(offset * sfreq))
        if end_sample > start_sample:
            annotations.append(Annotation(onset=start_sample, offset=end_sample))
    return annotations


def build_band_maps(data: np.ndarray, sfreq: float, bands: Sequence[Tuple[str, Optional[float], Optional[float]]]) -> List[np.ndarray]:
    maps: List[np.ndarray] = []
    for name, low, high in bands:
        if name == "raw" and low is None and high is None:
            maps.append(data)
        else:
            filtered = mne.filter.filter_data(
                data,
                sfreq=sfreq,
                l_freq=low,
                h_freq=high,
                verbose="ERROR",
                method="fir",
                phase="zero",
            )
            maps.append(filtered)
    return maps


def slice_windows(
    maps: Sequence[np.ndarray],
    annotations: Sequence[Annotation],
    sfreq: float,
    window_sec: float,
    stride_sec: float,
) -> Tuple[List[np.ndarray], List[dict]]:
    window_size = int(round(window_sec * sfreq))
    stride_size = max(int(round(stride_sec * sfreq)), 1)

    if window_size <= 0:
        raise ValueError("Window size must be positive.")

    total_samples = maps[0].shape[1]
    features: List[np.ndarray] = []
    metadata: List[dict] = []

    pointer = 0
    while pointer + window_size <= total_samples:
        window_slices = [band[:, pointer : pointer + window_size] for band in maps]
        window_has_seizure = any(
            ann.onset < pointer + window_size and ann.offset > pointer for ann in annotations
        )
        features.append(np.stack(window_slices, axis=0))
        metadata.append(
            {
                "window_start": pointer,
                "window_end": pointer + window_size,
                "target": int(window_has_seizure),
            }
        )
        pointer += stride_size
    return features, metadata


def flatten_feature_cube(cube: np.ndarray) -> np.ndarray:
    return cube.reshape(-1)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.rng_seed)

    all_records, _ = resolve_records(args.data_root, args.records_file, args.records_with_seizures)
    if args.max_files:
        all_records = all_records[: args.max_files]

    channel_template = load_channel_template(all_records, args.channel_list)
    print(f"[INFO] Using {len(channel_template)} EEG channels: {channel_template}")

    kept_features: List[np.ndarray] = []
    kept_metadata: List[dict] = []
    positive_count = 0
    negative_candidates: List[Tuple[np.ndarray, dict]] = []

    for idx, edf_path in enumerate(all_records, start=1):
        raw = read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        if args.highpass is not None or args.lowpass is not None:
            raw.filter(l_freq=args.highpass, h_freq=args.lowpass, verbose="ERROR")
        if args.notch_freq:
            raw.notch_filter(freqs=[args.notch_freq], verbose="ERROR")

        raw.pick_channels(channel_template, ordered=True)
        if args.target_sfreq and args.target_sfreq > 0 and not np.isclose(sfreq, args.target_sfreq):
            raw.resample(args.target_sfreq, npad="auto")
            sfreq = float(raw.info["sfreq"])

        data = raw.get_data().astype(np.float32)
        data -= data.mean(axis=1, keepdims=True)

        band_maps = build_band_maps(data, sfreq, DEFAULT_BANDS)

        annotation_file = edf_path.with_suffix(".seizures")
        annotations = parse_seizure_annotation(annotation_file, sfreq)

        cubes, meta_rows = slice_windows(
            band_maps,
            annotations,
            sfreq=sfreq,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
        )

        subject = edf_path.parent.name
        record_id = edf_path.stem

        seizure_in_file = sum(row["target"] for row in meta_rows)
        positive_count += seizure_in_file

        for cube, meta in zip(cubes, meta_rows):
            flattened = flatten_feature_cube(cube)
            meta_info = {
                "subject": subject,
                "record": record_id,
                "start_sec": meta["window_start"] / sfreq,
                "end_sec": meta["window_end"] / sfreq,
                "target": meta["target"],
            }
            if meta["target"] == 1:
                kept_features.append(flattened)
                kept_metadata.append(meta_info)
            else:
                negative_candidates.append((flattened, meta_info))

        raw.close()
        print(
            f"[INFO] Processed {edf_path.name} ({idx}/{len(all_records)}): "
            f"{len(meta_rows)} windows, {seizure_in_file} positives."
        )

    if args.balance_ratio and args.balance_ratio > 0 and positive_count > 0:
        max_negatives = int(round(positive_count * args.balance_ratio))
        if len(negative_candidates) > max_negatives:
            indices = rng.choice(len(negative_candidates), size=max_negatives, replace=False)
            selected = [negative_candidates[i] for i in indices]
        else:
            selected = negative_candidates
    else:
        selected = negative_candidates

    for flattened, meta in selected:
        kept_features.append(flattened)
        kept_metadata.append(meta)

    if not kept_features:
        raise RuntimeError("No windows were collected; check the dataset paths and parameters.")

    feature_matrix = np.stack(kept_features, axis=0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    feature_columns = [f"f{i:05d}" for i in range(scaled_features.shape[1])]
    feature_df = pd.DataFrame(scaled_features, columns=feature_columns)
    metadata_df = pd.DataFrame(kept_metadata)
    combined_df = pd.concat([metadata_df, feature_df], axis=1)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Wrote {len(combined_df)} windows to {args.output_csv}.")

    args.preprocessor_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "channels": channel_template,
            "bands": DEFAULT_BANDS,
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
        },
        args.preprocessor_output,
    )
    print(f"[INFO] Saved preprocessing artifact to {args.preprocessor_output}.")

    summary = {
        "total_windows": int(len(combined_df)),
        "positive_windows": int(metadata_df["target"].sum()),
        "negative_windows": int(len(combined_df) - int(metadata_df["target"].sum())),
        "channels": channel_template,
        "bands": DEFAULT_BANDS,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "target_sfreq": args.target_sfreq,
        "balance_ratio": args.balance_ratio,
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_json.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[INFO] Saved preprocessing summary to {args.metadata_json}.")


if __name__ == "__main__":
    main()
