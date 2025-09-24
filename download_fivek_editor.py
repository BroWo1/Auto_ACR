#!/usr/bin/env python3
"""Download MIT-Adobe FiveK raw/target pairs and index them as JSONL."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Iterable, Optional

if platform.system() == "Darwin":
    # Prevent "libomp already initialized" aborts triggered by torch DataLoader.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from torch.utils.data import DataLoader

MITAboveFiveK = None  # populated at runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download MIT-Adobe FiveK DNGs and targeted 16-bit TIFF(s) for the selected "
            "editor(s), then emit a JSONL index of (raw, target) pairs."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory that will contain MITAboveFiveK/",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "val", "test", "debug"],
        help="Dataset splits to download and index",
    )
    parser.add_argument(
        "--experts",
        type=str,
        default="c",
        help="Editors to download, e.g. 'c' or 'a,c,e' (case-insensitive)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader/download workers",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("fivek_editor_index.jsonl"),
        help="Path to the JSONL file to write",
    )
    parser.add_argument(
        "--dataset-repo",
        type=Path,
        default=None,
        help=(
            "Path to the cloned mit-adobe-fivek dataset repository. If omitted, the script "
            "looks for a sibling 'dataset/fivek.py'."
        ),
    )
    return parser.parse_args()


def load_dataset_class(dataset_repo: Optional[Path]):
    global MITAboveFiveK

    base_path = dataset_repo.expanduser() if dataset_repo is not None else Path(__file__).resolve().parent
    search_path = base_path.resolve()

    if not search_path.exists():
        raise SystemExit(f"Dataset repo path does not exist: {search_path}")

    candidate_paths = [search_path]
    dataset_subdir = search_path / "dataset"
    if dataset_subdir.exists():
        candidate_paths.append(dataset_subdir.resolve())

    for path in candidate_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        module = importlib.import_module("dataset.fivek")
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None)
        if missing and missing not in {"dataset", "dataset.fivek"}:
            raise SystemExit(
                f"Failed to import dataset dependency '{missing}'. Install it in your environment and retry."
            ) from exc
        raise SystemExit(
            "Unable to import 'dataset.fivek'. Provide --dataset-repo pointing to the "
            "mit-adobe-fivek repository (git clone https://github.com/yuukicammy/mit-adobe-fivek-dataset)."
        ) from exc

    MITAboveFiveK = getattr(module, "MITAboveFiveK")


def ensure_download(root: str, split: str, experts: list[str], workers: int) -> MITAboveFiveK:
    """Instantiate MITAboveFiveK with download enabled for the requested split."""
    dataset = MITAboveFiveK(
        root=root,
        split=split,
        download=True,
        experts=experts,
        download_workers=workers,
    )
    return dataset


def iterate_items(dataset: MITAboveFiveK, workers: int) -> Iterable[dict]:
    """Iterate over dataset items via DataLoader (must use batch_size=None)."""
    loader = DataLoader(dataset, batch_size=None, num_workers=workers)
    for sample in loader:
        yield sample


def main() -> None:
    args = parse_args()
    experts = [chunk.strip().lower() for chunk in args.experts.split(",") if chunk.strip()]

    load_dataset_class(args.dataset_repo)

    output_path = args.index
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as outfile:
        for split in args.splits:
            print(f"[fivek] split={split} experts={experts} → downloading…")
            dataset = ensure_download(args.root, split, experts, args.workers)

            print(f"[fivek] building index for split={split} …")
            for item in iterate_items(dataset, workers=args.workers):
                basename = item["basename"]
                dng_path = str(item["files"]["dng"])
                tiffs = item["files"].get("tiff16", {})

                for expert_key, tif_path in tiffs.items():
                    record = {
                        "split": split,
                        "id": item["id"],
                        "basename": basename,
                        "expert": expert_key,
                        "dng": dng_path,
                        "tif16": str(tif_path),
                        "camera": item.get("camera", {}),
                        "categories": item.get("categories", {}),
                        "license": item.get("license"),
                    }
                    outfile.write(json.dumps(record) + "\n")

            print(f"[fivek] done split={split}")

    print(f"[fivek] index written → {output_path.resolve()}")


if __name__ == "__main__":
    main()
