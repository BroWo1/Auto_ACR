#!/usr/bin/env python3
"""
Gradio front end for Auto ACR Autopilot rendering.

Users can upload a RAW (DNG) file, pick a base checkpoint plus optional LoRA
adapter, choose the output format, and preview/download the rendered result.
"""

from __future__ import annotations

import sys
import uuid
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run.predict_and_render import (  # noqa: E402
    SourceMetadata,
    build_transform,
    extract_source_metadata,
    load_lora_weights,
    load_model,
    predict_params,
    read_raw_linear_prophoto,
    read_raw_to_preview,
    select_device,
)
from scripts.fivek_tif_to_xmp import (  # noqa: E402
    edit_layer_realunits,
    prophoto_to_srgb_preview,
)

torch.set_grad_enabled(False)

DEFAULT_BACKBONE = "vit_small_patch16_224"
DEFAULT_IMAGE_SIZE = 224
USE_IMAGENET_NORM = False
TIFF_BIT_DEPTH = 16
OUTPUT_ROOT = ROOT / "outputs" / "gradio"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_CACHE: Dict[Tuple[Path, Optional[Path]], Tuple[torch.nn.Module, object, torch.device]] = {}


def _ascii_tag(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.encode("ascii", "ignore").decode("ascii").strip() or None


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def list_checkpoints() -> Tuple[str, ...]:
    choices = []
    search_roots = [
        ROOT / "run",
        ROOT / "run" / "checkpoints",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.pt")):
            try:
                rel = path.relative_to(ROOT).as_posix()
                if rel not in choices:
                    choices.append(rel)
            except ValueError:
                continue
    return tuple(choices)


def list_lora_adapters() -> Tuple[str, ...]:
    choices = []
    search_roots = [
        ROOT / "run" / "lora",
        ROOT / "outputs" / "lora",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.pt")):
            try:
                rel = path.relative_to(ROOT).as_posix()
                if rel not in choices:
                    choices.append(rel)
            except ValueError:
                continue
    return tuple(choices)


def resolve_checkpoint_path(choice: Optional[str], upload_path: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    if upload_path:
        path = Path(upload_path)
        if not path.exists():
            return None, f"Uploaded checkpoint not found: {path}"
        return path, None
    if choice and str(choice).strip():
        try:
            path = resolve_path(str(choice).strip())
        except Exception as exc:
            return None, f"Unable to resolve checkpoint path: {exc}"
        if not path.exists():
            return None, f"Checkpoint not found: {path}"
        return path, None
    return None, "Please select or upload a checkpoint (.pt) file."


def resolve_lora_path(choice: Optional[str], upload_path: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    if upload_path:
        path = Path(upload_path)
        if not path.exists():
            return None, f"Uploaded LoRA checkpoint not found: {path}"
        return path, None
    if choice and str(choice).strip():
        try:
            path = resolve_path(str(choice).strip())
        except Exception as exc:
            return None, f"Unable to resolve LoRA path: {exc}"
        if not path.exists():
            return None, f"LoRA checkpoint not found: {path}"
        return path, None
    return None, None


def get_model_artifacts(checkpoint_path: Path, lora_path: Optional[Path]):
    key = (checkpoint_path.resolve(), lora_path.resolve() if lora_path else None)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    device = select_device("auto")
    model = load_model(checkpoint_path, DEFAULT_BACKBONE, device)
    if lora_path is not None:
        load_lora_weights(model, lora_path, device)
    transform = build_transform(DEFAULT_IMAGE_SIZE, USE_IMAGENET_NORM)
    MODEL_CACHE[key] = (model, transform, device)
    return MODEL_CACHE[key]


def save_prophoto_tiff(
    edited: torch.Tensor,
    output_path: Path,
    metadata: Optional[SourceMetadata],
    bit_depth: int = TIFF_BIT_DEPTH,
    icc_profile_path: Optional[Path] = None,
) -> None:
    edited_cpu = edited.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    gamma = 1.8
    encoded = np.power(np.clip(edited_cpu, 0, 1), 1.0 / gamma)

    if bit_depth == 16:
        tiff_data = (np.clip(encoded * 65535.0, 0.0, 65535.0)).astype(np.uint16)
    else:
        tiff_data = (np.clip(encoded * 255.0, 0.0, 255.0)).astype(np.uint8)

    icc_bytes = None
    profile_path = icc_profile_path
    if profile_path is None:
        default_profile = ROOT / "ProPhoto.icm"
        if default_profile.exists():
            profile_path = default_profile
    if profile_path is not None and profile_path.exists():
        icc_bytes = profile_path.read_bytes()

    extratags = []
    software_tag = _ascii_tag("Auto_ACR Autopilot (Gradio)")
    if software_tag:
        extratags.append((305, "s", len(software_tag) + 1, software_tag + "\x00", False))
    if metadata:
        make_tag = _ascii_tag(metadata.make)
        model_tag = _ascii_tag(metadata.model)
        datetime_tag = _ascii_tag(metadata.datetime)
        if make_tag:
            extratags.append((271, "s", len(make_tag) + 1, make_tag + "\x00", False))
        if model_tag:
            extratags.append((272, "s", len(model_tag) + 1, model_tag + "\x00", False))
        if datetime_tag:
            extratags.append((306, "s", len(datetime_tag) + 1, datetime_tag + "\x00", False))
        if metadata.orientation is not None:
            extratags.append((274, "H", 1, int(metadata.orientation), False))

    try:
        with tifffile.TiffWriter(output_path) as tif:
            tif.write(
                tiff_data,
                photometric="rgb",
                contiguous=True,
                metadata=None,
                iccprofile=icc_bytes,
                extratags=extratags,
            )
    except Exception:
        with tifffile.TiffWriter(output_path) as tif:
            tif.write(
                tiff_data,
                photometric="rgb",
                contiguous=True,
                metadata=None,
                iccprofile=icc_bytes,
            )


def make_summary(params_denorm: Dict[str, object]) -> str:
    temp = params_denorm.get("Temperature", 0.0)
    tint = params_denorm.get("Tint", 0.0)
    exposure = params_denorm.get("Exposure2012", 0.0)
    contrast = params_denorm.get("Contrast2012", 0.0)
    highlights = params_denorm.get("Highlights2012", 0.0)
    shadows = params_denorm.get("Shadows2012", 0.0)
    return (
        f"Temperature {temp:.0f} K | Tint {tint:+.0f} | "
        f"Exposure {exposure:+.2f} EV | Contrast {contrast:+.0f}\n"
        f"Highlights {highlights:+.0f} | Shadows {shadows:+.0f}"
    )


def render_autopilot(
    raw_file,
    checkpoint_choice: Optional[str],
    checkpoint_upload: Optional[str],
    lora_choice: Optional[str],
    lora_upload: Optional[str],
    output_format: str,
):
    if raw_file is None:
        return None, None, None, "Upload a RAW (.dng) file to begin."

    checkpoint_path, checkpoint_error = resolve_checkpoint_path(checkpoint_choice, checkpoint_upload)
    if checkpoint_path is None:
        return None, None, None, checkpoint_error or "Checkpoint selection required."

    lora_path, lora_error = resolve_lora_path(lora_choice, lora_upload)
    if lora_error:
        return None, None, None, lora_error

    try:
        model, transform, device = get_model_artifacts(checkpoint_path, lora_path)
    except Exception as exc:
        traceback.print_exc()
        return None, None, None, f"Failed to load model: {exc}"

    raw_path = Path(raw_file.name)

    try:
        preview_rgb = read_raw_to_preview(raw_path, long_side=512)
    except Exception as exc:
        traceback.print_exc()
        return None, None, None, f"Failed to read RAW preview: {exc}"

    pil_preview = Image.fromarray(preview_rgb, mode="RGB")

    try:
        predictions = predict_params(model, pil_preview, transform, device)
    except Exception as exc:
        traceback.print_exc()
        return None, None, None, f"Inference failed: {exc}"

    params_denorm = predictions["params_denorm"]

    try:
        raw_linear = read_raw_linear_prophoto(raw_path)
    except Exception as exc:
        traceback.print_exc()
        return None, None, None, f"Failed to read RAW for rendering: {exc}"

    try:
        edited = edit_layer_realunits(raw_linear, params_denorm)
    except Exception as exc:
        traceback.print_exc()
        return None, None, None, f"Failed to apply edits: {exc}"

    preview01 = np.clip(prophoto_to_srgb_preview(edited), 0.0, 1.0)
    output_preview = (preview01 * 255.0 + 0.5).astype(np.uint8)

    metadata = extract_source_metadata(raw_path)

    session_dir = OUTPUT_ROOT / f"session_{uuid.uuid4().hex}"
    session_dir.mkdir(parents=True, exist_ok=True)

    fmt = (output_format or "tiff").strip().lower()
    if fmt not in {"tiff", "jpeg"}:
        fmt = "tiff"

    if fmt == "jpeg":
        output_path = session_dir / f"{raw_path.stem}_autopilot.jpg"
        pil_output = Image.fromarray(output_preview, mode="RGB")
        save_kwargs: Dict[str, object] = {"quality": 95}
        if metadata and metadata.exif_bytes:
            save_kwargs["exif"] = metadata.exif_bytes
        pil_output.save(output_path, format="JPEG", **save_kwargs)
    else:
        output_path = session_dir / f"{raw_path.stem}_autopilot.tif"
        save_prophoto_tiff(edited, output_path, metadata, bit_depth=TIFF_BIT_DEPTH)

    summary = make_summary(params_denorm)
    return preview_rgb, output_preview, str(output_path), summary


def build_ui() -> gr.Blocks:
    checkpoints = list_checkpoints()
    default_checkpoint = checkpoints[0] if checkpoints else ""

    lora_choices = list_lora_adapters()
    lora_dropdown_choices = [""] + list(lora_choices)
    default_lora = lora_choices[0] if lora_choices else ""

    with gr.Blocks(title="Auto ACR Autopilot") as demo:
        gr.Markdown(
            "## Auto ACR Autopilot Preview\n"
            "Upload a RAW (.dng) file, choose a base checkpoint and optional LoRA adapter, "
            "then render either a ProPhoto TIFF or sRGB JPEG."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                raw_input = gr.File(label="RAW input (.dng)", file_types=[".dng"])
                summary = gr.Textbox(label="Predicted sliders", interactive=False, lines=3)
                download = gr.File(label="Download rendered file")

            with gr.Column(scale=2):
                with gr.Row():
                    checkpoint_dropdown = gr.Dropdown(
                        choices=checkpoints,
                        value=default_checkpoint,
                        label="Base checkpoint (.pt)",
                        allow_custom_value=True,
                        scale=4,
                    )
                    checkpoint_upload = gr.UploadButton(
                        "Upload",
                        file_types=[".pt"],
                        file_count="single",
                        type="filepath",
                        variant="secondary",
                        scale=1,
                    )
                with gr.Row():
                    lora_dropdown = gr.Dropdown(
                        choices=lora_dropdown_choices,
                        value=default_lora,
                        label="LoRA adapter (optional)",
                        allow_custom_value=True,
                        scale=4,
                    )
                    lora_upload = gr.UploadButton(
                        "Upload",
                        file_types=[".pt"],
                        file_count="single",
                        type="filepath",
                        variant="secondary",
                        scale=1,
                    )
                output_format = gr.Radio(
                    choices=["TIFF", "JPEG"],
                    value="JPEG",
                    label="Output format",
                )
                run_button = gr.Button("Predict & Render", variant="primary")

        with gr.Row():
            input_preview = gr.Image(label="Input preview", type="numpy", height=360)
            output_image = gr.Image(label="Rendered preview", type="numpy", height=360)

        outputs = [input_preview, output_image, download, summary]
        run_button.click(
            fn=render_autopilot,
            inputs=[
                raw_input,
                checkpoint_dropdown,
                checkpoint_upload,
                lora_dropdown,
                lora_upload,
                output_format,
            ],
            outputs=outputs,
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=True)
