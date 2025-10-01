#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Randomly split a dataset manifest JSONL into training, validation, and testing splits.

Usage:
  python scripts/split_manifest.py \
    --manifest outputs/dataset_manifest.jsonl \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42

The script reads all entries, shuffles them randomly, and overwrites the 'split' field
in each entry according to the specified ratios. The modified manifest is written back
to the same file (or a new file if --output is specified).
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict


def main():
    parser = argparse.ArgumentParser(
        description="Randomly split manifest entries into train/val/test splits"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to input manifest JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output manifest (default: overwrites input)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction for training split (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction for validation split (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction for testing split (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Validate ratios sum to 1.0
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        print(f"[ERROR] Ratios must sum to 1.0, got {total}")
        return

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest file not found: {manifest_path}")
        return

    output_path = Path(args.output) if args.output else manifest_path

    # Load all records
    records: List[Dict] = []
    print(f"[info] Loading manifest from {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"[warn] Skipping invalid line: {e}")

    if not records:
        print("[ERROR] No valid records found in manifest")
        return

    print(f"[info] Loaded {len(records)} records")

    # Shuffle with seed
    random.seed(args.seed)
    random.shuffle(records)

    # Calculate split indices
    n = len(records)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    # Remaining goes to test to ensure all records are assigned
    n_test = n - n_train - n_val

    print(f"[info] Split sizes: train={n_train}, val={n_val}, test={n_test}")

    # Assign splits
    for i, record in enumerate(records):
        if i < n_train:
            record["split"] = "training"
        elif i < n_train + n_val:
            record["split"] = "validation"
        else:
            record["split"] = "testing"

    # Write back to manifest
    print(f"[info] Writing split manifest to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"[info] Done! Manifest written with {len(records)} entries")
    print(f"[info] Training: {n_train}, Validation: {n_val}, Testing: {n_test}")


if __name__ == "__main__":
    main()