#!/usr/bin/env python3
"""Fit ACR-style parameters for RAW/TIFF pairs and export JSON (and optional previews)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import rawpy
import torch
import imageio.v3 as iio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fivek_tif_to_xmp import fit_params_for_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True,
                        help="Path to a RAW file or directory containing .dng files")
    parser.add_argument("--tif", type=Path, required=True,
                        help="Path to a TIFF file or directory containing edited .tif files")
    parser.add_argument("--out-json", type=Path, required=True,
                        help="Directory where JSON parameter files will be written")
    parser.add_argument("--out-edited-preview", type=Path, default=None,
                        help="Optional directory for edited sRGB preview PNGs")
    parser.add_argument("--out-raw-preview", type=Path, default=None,
                        help="Optional directory for RAW sRGB previews (model inputs)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional path (file or directory) to write a manifest JSONL compatible with training")
    parser.add_argument("--split", default="training", help="Split name to record in the manifest")
    parser.add_argument("--expert", default="style", help="Expert label for manifest entries")
    parser.add_argument("--device", default="auto",
                        help="Torch device to run the fitting on (auto|cuda|cpu|mps)")
    parser.add_argument("--long-side", type=int, default=512,
                        help="Resize long side while fitting (smaller is faster)")
    parser.add_argument("--steps", type=int, default=600,
                        help="Gradient steps for the fit")
    parser.add_argument("--lr", type=float, default=5e-2,
                        help="Learning rate for the fit optimizer")
    parser.add_argument("--curve-knots", type=int, default=8,
                        help="Tone-curve knot count for the optimizer")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def collect_pairs(raw_root: Path, tif_root: Path) -> List[Tuple[Path, Path]]:
    if raw_root.is_file() and tif_root.is_file():
        return [(raw_root, tif_root)]

    if not raw_root.is_dir() or not tif_root.is_dir():
        raise ValueError("When providing directories both --raw and --tif must be directories")

    raw_files = {p.stem: p for p in raw_root.rglob("*.dng")}
    tif_files = {}
    for pattern in ("*.tif", "*.tiff"):
        for p in tif_root.rglob(pattern):
            tif_files.setdefault(p.stem, p)
    pairs: List[Tuple[Path, Path]] = []
    for stem, raw_path in sorted(raw_files.items()):
        tif_path = tif_files.get(stem)
        if tif_path is None:
            continue
        pairs.append((raw_path, tif_path))
    if not pairs:
        raise RuntimeError("No RAW/TIFF pairs with matching stems were found")
    return pairs


def ensure_dir(path: Optional[Path]) -> None:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


def write_json(params_denorm: dict, params_norm: dict, out_path: Path) -> None:
    payload = {
        "format": "acr-autopilot-v1",
        "params_denorm": {
            k: (float(v) if k != "ToneCurve" else list(map(float, v)))
            for k, v in params_denorm.items()
        },
        "params_norm": {
            k: (float(v) if k != "ToneCurve" else list(map(float, v)))
            for k, v in params_norm.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_preview(preview01: np.ndarray, out_path: Path) -> None:
    arr8 = (np.clip(preview01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    iio.imwrite(out_path, arr8)


def build_raw_preview(raw_path: Path, long_side: int) -> np.ndarray:
    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess(
            output_bps=8,
            no_auto_bright=True,
            gamma=(2.222, 4.5),
            output_color=rawpy.ColorSpace.sRGB,
            use_camera_wb=True,
        )
    h, w = rgb.shape[:2]
    if max(h, w) > long_side:
        scale = long_side / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return rgb


def select_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    pairs = collect_pairs(args.raw, args.tif)
    ensure_dir(args.out_json)
    ensure_dir(args.out_edited_preview)
    ensure_dir(args.out_raw_preview)
    manifest_path: Optional[Path] = args.manifest
    if manifest_path is not None and manifest_path.is_dir():
        manifest_path = manifest_path / "style_manifest.jsonl"
    if manifest_path is not None and args.out_raw_preview is None:
        print("[warn] --manifest provided but --out-raw-preview missing; manifest entries will be skipped")
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_records: List[dict] = []

    for raw_path, tif_path in pairs:
        stem = raw_path.stem
        json_path = args.out_json / f"{stem}.json"
        raw_preview_path = args.out_raw_preview / f"{stem}_raw.png" if args.out_raw_preview is not None else None
        edited_preview_path = args.out_edited_preview / f"{stem}_edited.png" if args.out_edited_preview is not None else None

        need_fit = args.overwrite or not json_path.exists()
        if json_path.exists() and not args.overwrite:
            print(f"[skip] {stem}: JSON exists")
        else:
            print(f"[fit] {stem}")

        if need_fit:
            params_denorm, params_norm, edited_preview = fit_params_for_pair(
                raw_path=raw_path,
                tif_path=tif_path,
                device=device,
                long_side=args.long_side,
                steps=args.steps,
                lr=args.lr,
                curve_knots=args.curve_knots,
            )

            write_json(params_denorm, params_norm, json_path)

            if edited_preview_path is not None:
                write_preview(edited_preview, edited_preview_path)
        if manifest_path is not None and raw_preview_path is not None:
            if not raw_preview_path.exists():
                raw_preview = build_raw_preview(raw_path, args.long_side)
                ensure_dir(raw_preview_path.parent)
                iio.imwrite(raw_preview_path, raw_preview)

        if manifest_path is not None and raw_preview_path is not None and raw_preview_path.exists():
            manifest_records.append({
                "id": stem,
                "split": args.split,
                "expert": args.expert,
                "raw": str(raw_path.resolve()),
                "tif": str(tif_path.resolve()),
                "preview": str(raw_preview_path.resolve()),
                "data": str(json_path.resolve()),
            })

    if manifest_path is not None and manifest_records:
        base = manifest_path.parent.resolve()

        def _rel(path_str: str) -> str:
            path = Path(path_str)
            try:
                return str(path.resolve().relative_to(base))
            except ValueError:
                return str(path.resolve())

        with manifest_path.open("w", encoding="utf-8") as mf:
            for record in manifest_records:
                mf.write(json.dumps({
                    "id": record["id"],
                    "split": record["split"],
                    "expert": record["expert"],
                    "raw": _rel(record["raw"]),
                    "tif": _rel(record["tif"]),
                    "preview": _rel(record["preview"]),
                    "data": _rel(record["data"]),
                }) + "\n")
        print(f"[info] Manifest written → {manifest_path.resolve()}")

    print("[done] Processed", len(pairs), "pair(s)")


if __name__ == "__main__":
    main()
