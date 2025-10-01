#!/usr/bin/env python3
"""
Inference script: Apply trained AutoPilot model to RAW images and generate XMP sidecars.

Usage:
  python run/inference.py \
    --checkpoint run/checkpoints/best.pt \
    --input path/to/image.dng \
    --output outputs/predictions/

  # Batch process a directory
  python run/inference.py \
    --checkpoint run/checkpoints/best.pt \
    --input data/raw_images/ \
    --output outputs/predictions/ \
    --batch
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import rawpy
import cv2
import imageio.v3 as iio
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from model.autopilot import AutoPilotRegressor, BackboneConfig
from params.ranges import SCALAR_NAMES, denormalize_scalars, denormalize_tone_curve

# ImageNet normalization (if model was trained with it)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--input", type=Path, required=True, help="Input RAW file or directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions"), help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Process all RAW files in input directory")
    parser.add_argument("--image-size", type=int, default=224, help="Model input size")
    parser.add_argument("--imagenet-norm", action="store_true", help="Apply ImageNet normalization (must match training)")
    parser.add_argument("--device", default="auto", help="Device: auto|cuda|cpu|mps")
    parser.add_argument("--backbone", default="vit_small_patch16_224", help="Backbone architecture (must match training)")
    parser.add_argument("--export-xmp", action="store_true", default=True, help="Export XMP sidecars")
    parser.add_argument("--export-json", action="store_true", help="Export JSON parameters")
    parser.add_argument("--export-preview", action="store_true", help="Export sRGB preview with predicted edits")
    parser.add_argument("--no-export-xmp", dest="export_xmp", action="store_false", help="Disable XMP export")
    return parser.parse_args()


def select_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def load_model(checkpoint_path: Path, backbone_name: str, device: torch.device) -> AutoPilotRegressor:
    """Load trained model from checkpoint."""
    backbone_cfg = BackboneConfig(
        name=backbone_name,
        pretrained=False,  # Using trained weights
        drop_path_rate=0.1,
    )
    model = AutoPilotRegressor(backbone=backbone_cfg, metadata_encoder=None, fusion="film")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print(f"[info] Loaded model from {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"[info] Checkpoint epoch: {checkpoint['epoch']}")
    if "val_stats" in checkpoint and checkpoint["val_stats"]:
        print(f"[info] Validation loss: {checkpoint['val_stats'].get('loss', 'N/A')}")

    return model


def build_transform(image_size: int, imagenet_norm: bool) -> T.Compose:
    """Build preprocessing transform for model input."""
    ops = [
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
    ]
    if imagenet_norm:
        ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(ops)


def read_raw_to_preview(raw_path: Path, long_side: int = 512) -> np.ndarray:
    """
    Read RAW file and generate sRGB preview (unedited baseline).
    Returns HxWx3 uint8 array.
    """
    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess(
            output_bps=8,
            no_auto_bright=True,
            gamma=(2.222, 4.5),  # sRGB gamma
            output_color=rawpy.ColorSpace.sRGB,
            use_camera_wb=True,
        )

    h, w = rgb.shape[:2]
    if max(h, w) > long_side:
        scale = long_side / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return rgb


def read_raw_linear_prophoto(raw_path: Path, long_side: int = 512) -> torch.Tensor:
    """
    Read RAW in linear ProPhoto RGB for applying edits.
    Returns [1,3,H,W] float tensor.
    """
    with rawpy.imread(str(raw_path)) as raw:
        rgb16 = raw.postprocess(
            output_bps=16,
            no_auto_bright=True,
            gamma=(1, 1),
            output_color=rawpy.ColorSpace.ProPhoto,
            use_camera_wb=True,
        )

    h, w = rgb16.shape[:2]
    if max(h, w) > long_side:
        scale = long_side / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        rgb16 = cv2.resize(rgb16, (new_w, new_h), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(rgb16).permute(2, 0, 1).float().unsqueeze(0) / 65535.0
    return tensor


@torch.no_grad()
def predict_params(model: AutoPilotRegressor, image: Image.Image, transform: T.Compose, device: torch.device) -> Dict[str, object]:
    """
    Run inference and return denormalized parameters.
    """
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    outputs = model(input_tensor)["params_norm"]

    # Extract predictions
    params_norm = {}
    for name in SCALAR_NAMES:
        params_norm[name] = outputs[name].squeeze().cpu().item()
    params_norm["ToneCurve"] = outputs["ToneCurve"].squeeze().cpu().numpy()

    # Denormalize
    params_denorm = denormalize_scalars(params_norm)
    params_denorm["ToneCurve"] = denormalize_tone_curve(params_norm["ToneCurve"])

    return {"params_norm": params_norm, "params_denorm": params_denorm}


def export_xmp(params_denorm: Dict[str, object], output_path: Path) -> None:
    """Export ACR-compatible XMP sidecar."""
    from scripts.fivek_tif_to_xmp import xmp_from_params

    xmp_str = xmp_from_params(
        params_denorm,
        camera_profile="Adobe Color",
        process_version="11.0",
        version=None
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xmp_str, encoding="utf-8")
    print(f"[saved] XMP → {output_path}")


def export_json_params(params_denorm: Dict[str, object], params_norm: Dict[str, object], output_path: Path) -> None:
    """Export parameters as JSON."""
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[saved] JSON → {output_path}")


def export_preview_with_edits(raw_path: Path, params_denorm: Dict[str, object], output_path: Path) -> None:
    """Apply predicted edits and export sRGB preview."""
    from scripts.fivek_tif_to_xmp import edit_layer_realunits, prophoto_to_srgb_preview

    # Load RAW in linear ProPhoto
    raw_linear = read_raw_linear_prophoto(raw_path, long_side=512)

    # Apply edits
    edited = edit_layer_realunits(raw_linear, params_denorm)

    # Convert to sRGB preview
    preview_srgb01 = prophoto_to_srgb_preview(edited)
    preview_uint8 = (np.clip(preview_srgb01 * 255.0, 0, 255)).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, preview_uint8)
    print(f"[saved] Preview → {output_path}")


def process_single_image(
    raw_path: Path,
    model: AutoPilotRegressor,
    transform: T.Compose,
    device: torch.device,
    output_dir: Path,
    export_xmp: bool,
    export_json: bool,
    export_preview: bool,
) -> None:
    """Process a single RAW image."""
    print(f"\n[processing] {raw_path.name}")

    # Read RAW as sRGB preview for model input
    preview_rgb = read_raw_to_preview(raw_path, long_side=512)
    pil_image = Image.fromarray(preview_rgb, mode="RGB")

    # Predict parameters
    result = predict_params(model, pil_image, transform, device)
    params_denorm = result["params_denorm"]
    params_norm = result["params_norm"]

    # Print key predictions
    print(f"  Temperature: {params_denorm['Temperature']:.0f}K")
    print(f"  Exposure: {params_denorm['Exposure2012']:+.2f} EV")
    print(f"  Contrast: {params_denorm['Contrast2012']:+.0f}")

    # Export outputs
    stem = raw_path.stem

    if export_xmp:
        xmp_path = output_dir / "xmps" / f"{stem}.xmp"
        export_xmp(params_denorm, xmp_path)

    if export_json:
        json_path = output_dir / "json" / f"{stem}.json"
        export_json_params(params_denorm, params_norm, json_path)

    if export_preview:
        preview_path = output_dir / "previews" / f"{stem}_edited.png"
        export_preview_with_edits(raw_path, params_denorm, preview_path)


def main():
    args = parse_args()
    device = select_device(args.device)

    # Load model
    model = load_model(args.checkpoint, args.backbone, device)
    transform = build_transform(args.image_size, args.imagenet_norm)

    # Collect input files
    input_path = args.input
    if args.batch:
        if not input_path.is_dir():
            raise ValueError(f"--batch requires a directory, got: {input_path}")
        raw_files = sorted(input_path.glob("*.dng")) + sorted(input_path.glob("*.DNG"))
        if not raw_files:
            raise ValueError(f"No .dng files found in {input_path}")
        print(f"[info] Found {len(raw_files)} RAW files")
    else:
        if not input_path.is_file():
            raise ValueError(f"Input file not found: {input_path}")
        raw_files = [input_path]

    # Process each file
    for raw_path in raw_files:
        try:
            process_single_image(
                raw_path=raw_path,
                model=model,
                transform=transform,
                device=device,
                output_dir=args.output,
                export_xmp=args.export_xmp,
                export_json=args.export_json,
                export_preview=args.export_preview,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {raw_path.name}: {e}")
            continue

    print(f"\n[done] Processed {len(raw_files)} images")
    print(f"[info] Outputs saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
