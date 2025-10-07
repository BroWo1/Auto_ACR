#!/usr/bin/env python3
"""Convert TIFF images from Adobe RGB to ProPhoto (ROMM) colorspace."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import tifffile


ADOBE_RGB_TO_XYZ = np.array([
    [0.5767309, 0.1855540, 0.1881852],
    [0.2973769, 0.6273491, 0.0752741],
    [0.0270343, 0.0706872, 0.9911085],
], dtype=np.float32)

PROPHOTO_TO_XYZ = np.array([
    [0.7976749, 0.1351917, 0.0313534],
    [0.2880402, 0.7118741, 0.0000857],
    [0.0000000, 0.0000000, 0.8252100],
], dtype=np.float32)

XYZ_TO_PROPHOTO = np.linalg.inv(PROPHOTO_TO_XYZ)

BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
], dtype=np.float32)

BRADFORD_INV = np.linalg.inv(BRADFORD)

D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
D50 = np.array([0.96422, 1.00000, 0.82521], dtype=np.float32)


def build_adobe_to_prophoto_matrix() -> np.ndarray:
    # Adapt from D65 to D50 using Bradford
    cone_src = BRADFORD @ D65
    cone_dst = BRADFORD @ D50
    adapt = BRADFORD_INV @ np.diag(cone_dst / cone_src) @ BRADFORD
    return XYZ_TO_PROPHOTO @ adapt @ ADOBE_RGB_TO_XYZ


ADOBE_TO_PROPHOTO = build_adobe_to_prophoto_matrix()


def decode_adobe_gamma(encoded: np.ndarray) -> np.ndarray:
    return np.power(np.clip(encoded, 0.0, 1.0), 2.19921875)


def encode_prophoto_gamma(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    thresh = 1.0 / 512.0
    encoded = np.where(linear < thresh, linear * 16.0, np.power(linear, 1.0 / 1.8))
    return np.clip(encoded, 0.0, 1.0)


def convert_image(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Only RGB images are supported")

    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        arr01 = arr.astype(np.float32) / max_value
        dtype = arr.dtype
    elif np.issubdtype(arr.dtype, np.floating):
        arr01 = np.clip(arr.astype(np.float32), 0.0, 1.0)
        dtype = np.float32
    else:
        raise TypeError(f"Unsupported dtype: {arr.dtype}")

    linear = decode_adobe_gamma(arr01)
    reshaped = linear.reshape(-1, 3)
    converted = reshaped @ ADOBE_TO_PROPHOTO.T
    converted = converted.reshape(linear.shape)
    encoded = encode_prophoto_gamma(converted)

    if np.issubdtype(dtype, np.integer):
        out = np.clip(encoded * max_value + 0.5, 0.0, max_value).astype(dtype)
    else:
        out = encoded.astype(dtype)
    return out


def iter_tiff_files(root: Path, pattern: str) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in {".tif", ".tiff"}:
            yield root
        return
    for path in sorted(root.rglob(pattern)):
        yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="TIFF file or directory of TIFFs")
    parser.add_argument("--output", type=Path, required=True, help="Output directory or file")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern for TIFF search when input is a directory")
    parser.add_argument("--icc-profile", type=Path, default=None,
                        help="Optional ProPhoto ICC profile to embed (defaults to repo ProPhoto.icm if present)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_output_path(input_path: Path, input_root: Path, output_base: Path) -> Path:
    if output_base.is_dir() or input_root.is_dir():
        rel = input_path.relative_to(input_root) if input_root.is_dir() else input_path.name
        if isinstance(rel, Path):
            rel_path = rel
        else:
            rel_path = Path(rel)
        return output_base / rel_path
    return output_base


def load_icc(profile_path: Path | None) -> bytes | None:
    if profile_path is not None:
        if profile_path.exists():
            return profile_path.read_bytes()
        print(f"[warn] ICC profile not found at {profile_path}; skipping embedding")
        return None
    default_profile = Path(__file__).resolve().parents[1] / "ProPhoto.icm"
    if default_profile.exists():
        return default_profile.read_bytes()
    return None


def process_file(path: Path, out_path: Path, icc_bytes: bytes | None, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        print(f"[skip] {path.name}: output exists")
        return

    data = tifffile.imread(path)
    converted = convert_image(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_path, converted, photometric="rgb", metadata=None, iccprofile=icc_bytes)
    print(f"[saved] {out_path}")


def main() -> None:
    args = parse_args()
    icc_bytes = load_icc(args.icc_profile)

    input_root = args.input if args.input.is_dir() else args.input.parent
    files = list(iter_tiff_files(args.input, args.pattern))
    if not files:
        raise SystemExit("No TIFF files found")

    for file_path in files:
        out_path = resolve_output_path(file_path, input_root, args.output)
        process_file(file_path, out_path, icc_bytes, args.overwrite)

    print(f"[done] Converted {len(files)} file(s)")


if __name__ == "__main__":
    main()
