#!/usr/bin/env python3
"""
Predict ACR parameters and render edited TIFF from RAW.

Usage:
  python run/predict_and_render.py \
    --checkpoint run/checkpoints/best.pt \
    --input path/to/image.dng \
    --output outputs/rendered/

  # Batch process
  python run/predict_and_render.py \
    --checkpoint run/checkpoints/best.pt \
    --input data/raw_images/ \
    --output outputs/rendered/ \
    --batch
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

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

# Add scripts to path for edit layer access
scripts_path = Path(__file__).parent.parent / "scripts"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

from fivek_tif_to_xmp import edit_layer_realunits, prophoto_to_srgb_preview

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--input", type=Path, required=True, help="Input RAW file or directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/rendered"), help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Process all RAW files in input directory")
    parser.add_argument("--image-size", type=int, default=224, help="Model input size")
    parser.add_argument("--imagenet-norm", action="store_true", help="Apply ImageNet normalization")
    parser.add_argument("--device", default="auto", help="Device: auto|cuda|cpu|mps")
    parser.add_argument("--backbone", default="vit_small_patch16_224", help="Backbone architecture")
    parser.add_argument("--tiff-bit-depth", type=int, choices=[8, 16], default=16, help="TIFF bit depth")
    parser.add_argument("--export-json", action="store_true", help="Also export predicted parameters as JSON")
    parser.add_argument("--render-resolution", type=int, default=2048, help="Max resolution for rendering (lower = faster)")
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
        pretrained=False,
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

    return model


def build_transform(image_size: int, imagenet_norm: bool) -> T.Compose:
    """Build preprocessing transform."""
    ops = [
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
    ]
    if imagenet_norm:
        ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(ops)


def read_raw_to_preview(raw_path: Path, long_side: int = 512) -> np.ndarray:
    """Read RAW and generate sRGB preview for model input."""
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


def read_raw_linear_prophoto(raw_path: Path, max_resolution: int = 2048) -> torch.Tensor:
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
    if max(h, w) > max_resolution:
        scale = max_resolution / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        rgb16 = cv2.resize(rgb16, (new_w, new_h), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(rgb16).permute(2, 0, 1).float().unsqueeze(0) / 65535.0
    return tensor


@torch.no_grad()
def predict_params(model: AutoPilotRegressor, image: Image.Image, transform: T.Compose, device: torch.device) -> Dict[str, object]:
    """Run inference and return denormalized parameters."""
    input_tensor = transform(image).unsqueeze(0).to(device)
    outputs = model(input_tensor)["params_norm"]

    params_norm = {}
    for name in SCALAR_NAMES:
        params_norm[name] = outputs[name].squeeze().cpu().item()
    params_norm["ToneCurve"] = outputs["ToneCurve"].squeeze().cpu().numpy()

    params_denorm = denormalize_scalars(params_norm)
    params_denorm["ToneCurve"] = denormalize_tone_curve(params_norm["ToneCurve"])

    return {"params_norm": params_norm, "params_denorm": params_denorm}


def apply_edits_and_save_tiff(
    raw_path: Path,
    params_denorm: Dict[str, object],
    output_path: Path,
    bit_depth: int = 16,
    max_resolution: int = 2048,
) -> None:
    """
    Apply predicted edits to RAW and save as TIFF in ProPhoto RGB.

    Args:
        raw_path: Path to DNG file
        params_denorm: Denormalized ACR parameters
        output_path: Output TIFF path
        bit_depth: 8 or 16 bit output
        max_resolution: Maximum resolution for rendering
    """
    # Read RAW in linear ProPhoto RGB
    print(f"  Reading RAW: {raw_path.name}")
    raw_linear = read_raw_linear_prophoto(raw_path, max_resolution=max_resolution)

    # Apply edits using differentiable edit layer
    print(f"  Applying edits...")
    edited = edit_layer_realunits(raw_linear, params_denorm)

    # Convert to numpy and encode with ProPhoto gamma
    edited_np = edited.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Gamma encode: linear → gamma 1.8 (ProPhoto standard)
    gamma = 1.8
    edited_gamma = np.power(np.clip(edited_np, 0, 1), 1.0 / gamma)

    # Quantize to target bit depth
    if bit_depth == 16:
        tiff_data = (np.clip(edited_gamma * 65535, 0, 65535)).astype(np.uint16)
    else:
        tiff_data = (np.clip(edited_gamma * 255, 0, 255)).astype(np.uint8)

    # Save TIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, tiff_data)
    print(f"[saved] TIFF → {output_path}")


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


def process_single_image(
    raw_path: Path,
    model: AutoPilotRegressor,
    transform: T.Compose,
    device: torch.device,
    output_dir: Path,
    tiff_bit_depth: int,
    render_resolution: int,
    export_json: bool,
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

    # Generate output paths
    stem = raw_path.stem
    tiff_path = output_dir / "tiff" / f"{stem}.tif"

    # Apply edits and save TIFF
    apply_edits_and_save_tiff(
        raw_path=raw_path,
        params_denorm=params_denorm,
        output_path=tiff_path,
        bit_depth=tiff_bit_depth,
        max_resolution=render_resolution,
    )

    # Optionally export JSON
    if export_json:
        json_path = output_dir / "json" / f"{stem}.json"
        export_json_params(params_denorm, params_norm, json_path)


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
        raw_files_set = set(input_path.glob("*.dng")) | set(input_path.glob("*.DNG"))
        raw_files = sorted(raw_files_set)
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
                tiff_bit_depth=args.tiff_bit_depth,
                render_resolution=args.render_resolution,
                export_json=args.export_json,
            )
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to process {raw_path.name}: {e}")
            traceback.print_exc()
            continue

    print(f"\n[done] Processed {len(raw_files)} images")
    print(f"[info] TIFFs saved to {(args.output / 'tiff').resolve()}")


if __name__ == "__main__":
    main()
