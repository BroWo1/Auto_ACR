# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Auto ACR is a machine learning system that learns Adobe Camera Raw (ACR) adjustments from expert-edited images. It consists of two main workflows:

1. **Data preparation**: Convert RAW/TIFF pairs to normalized parameters via differentiable optimization
2. **Model training**: Train a neural network to predict ACR parameters from RAW preview images

## Environment Setup

Create the conda environment:
```bash
conda env create -f environment.yml
conda activate auto_acr
```

Key dependencies: PyTorch, rawpy, imageio, opencv, tqdm, timm (via pip)

## Core Workflows

### 1. Data Preparation (RAW → Parameters)

**Script**: `scripts/fivek_tif_to_xmp.py`

Fits Camera Raw PV2012 parameters to RAW/TIFF pairs using differentiable optimization:

```bash
# Process MIT-Adobe FiveK dataset
python scripts/fivek_tif_to_xmp.py \
  --root data/MITAboveFiveK \
  --split training \
  --expert c \
  --device mps \
  --steps 800
```

**Key parameters**:
- `--split`: training, validation, testing, or debugging
- `--expert`: Expert editor (a, b, c, d, or e)
- `--device`: mps (Apple Silicon), cuda, or cpu
- `--steps`: Optimization iterations per image (default: 800)
- `--curve_knots`: Tone curve flexibility (default: 16)

**Outputs** (relative to `outputs/`):
- `xmps/`: XMP sidecar files for Adobe Camera Raw
- `preview/`: sRGB preview PNGs from RAW
- `data/`: JSON files with normalized parameters
- `dataset_manifest.jsonl`: Training manifest (auto-merges and rebuilds from existing files on each run)

The script uses a differentiable "mini-ACR" edit layer that models:
- White balance (Temperature, Tint) as channel gains
- Exposure as EV stops
- Contrast via tanh S-curve
- Regional adjustments (Highlights, Shadows, Whites, Blacks) using luminance masks
- Monotone tone curve via piecewise-linear interpolation of learned knots
- Vibrance/Saturation in YUV space

All operations happen in **linear ProPhoto RGB** for color accuracy.

### 2. Model Training

**Script**: `run/train_autopilot.py`

Trains `AutoPilotRegressor` to predict normalized ACR parameters from preview images.

**Parameter-only training (baseline):**
```bash
python run/train_autopilot.py \
  --manifest outputs/dataset_manifest.jsonl \
  --train-split training \
  --val-split validation \
  --batch-size 32 \
  --epochs 20 \
  --device auto
```

**Hybrid training with image-space loss (recommended):**
```bash
python run/train_autopilot.py \
  --manifest outputs/dataset_manifest.jsonl \
  --train-split training \
  --val-split validation \
  --batch-size 12 \
  --epochs 20 \
  --device auto \
  --use-edit-layer \
  --image-loss-weight 0.7 \
  --edit-resolution 512
```

**Key parameters**:
- `--manifest`: Path to JSONL manifest (contains all file paths)
- `--use-edit-layer`: Enable DifferentiableEditLayer for image-space loss
- `--image-loss-weight`: Weight for image loss vs param loss (default: 0.7)
- `--edit-resolution`: Resolution for applying edits (default: 512, lower = less memory)
- `--pretrained-backbone`: Use ImageNet pretrained weights
- `--amp`: Enable automatic mixed precision

**Outputs**:
- `run/checkpoints/checkpoint_epochXXX.pt`: Periodic checkpoints
- `run/checkpoints/best.pt`: Best validation checkpoint

## Architecture

### Model Structure (`model/autopilot.py`)

**AutoPilotRegressor** is a single-encoder multi-head regressor:

```
Image → ImageBackbone → [optional MetadataEncoder + Fusion] → { SlidersHead, ToneCurveHead }
```

1. **ImageBackbone** (`model/backbones.py`):
   - Wraps any timm encoder (ViT, ResNet, etc.)
   - Returns pooled features via token or mean pooling
   - Configurable via `BackboneConfig`

2. **MetadataEncoder** (optional):
   - Embeds EXIF metadata (ISO, shutter, aperture, camera model)
   - Currently unused in training but ready for future experiments

3. **Fusion** (optional):
   - FiLM: Feature-wise Linear Modulation
   - Concat: Concatenation + projection
   - Currently unused (metadata_encoder=None in training)

4. **Prediction Heads** (`model/heads.py`):
   - **SlidersHead**: Predicts 10 scalar parameters in [-1,1] via tanh
   - **ToneCurveHead**: Predicts monotone tone curve via softplus + cumsum normalization

### Parameter Normalization (`params/ranges.py`)

All parameters are normalized to [-1, 1] for training:
- **Temperature**: 2000-50000K → [-1, 1]
- **Tint**: -150 to 150 → [-1, 1]
- **Exposure2012**: -5 to 5 EV → [-1, 1]
- **Contrast/Highlights/Shadows/Whites/Blacks/Vibrance/Saturation**: -100 to 100 → [-1, 1]
- **ToneCurve**: 16 knots in [0, 1], enforced monotone with endpoints at 0 and 1

Use `normalize_scalars()` / `denormalize_scalars()` and `normalize_tone_curve()` / `denormalize_tone_curve()` for conversion.

## Data Format

### Dataset Manifest (JSONL)
Each line contains:
```json
{
  "id": "a0001-jmac_DSC1459",
  "split": "training",
  "expert": "c",
  "raw": "data/MITAboveFiveK/raw/.../a0001-jmac_DSC1459.dng",
  "tif": "data/MITAboveFiveK/processed/tiff16_c/.../a0001-jmac_DSC1459.tif",
  "preview": "outputs/preview/a0001-jmac_DSC1459.png",
  "data": "outputs/data/a0001-jmac_DSC1459.json"
}
```

**Note:** The manifest automatically merges with existing entries when running the fitting script multiple times. Entries with the same (id, split, expert) are updated.

### Parameter JSON
```json
{
  "format": "acr-autopilot-v1",
  "params_denorm": {
    "Temperature": 6500.0,
    "Tint": 0.0,
    "Exposure2012": 0.5,
    ...
    "ToneCurve": [0.0, 0.15, 0.3, ..., 1.0]
  },
  "params_norm": {
    "Temperature": 0.0,
    "Tint": 0.0,
    "Exposure2012": 0.1,
    ...
    "ToneCurve": [0.0, 0.15, 0.3, ..., 1.0]
  }
}
```

## Color Management

The codebase uses **linear ProPhoto RGB** internally for fitting and editing:
- RAWs are decoded to linear ProPhoto via rawpy
- Target TIFFs are gamma-decoded to linear ProPhoto
- Fitting happens in linear space for physically accurate optimization
- Preview outputs are converted to sRGB via ICC profiles (macOS) or matrix transform + Bradford CAT

**Key color functions** in `scripts/fivek_tif_to_xmp.py`:
- `read_raw_linear_prophoto()`: RAW → linear ProPhoto tensor
- `read_tif16()`: TIFF → linear ProPhoto tensor
- `prophoto_to_srgb_preview()`: Linear ProPhoto → sRGB [0,1] via ICC
- `_prophoto_encode_torch()` / `_prophoto_decode_torch()`: Gamma 1.8 conversion

## Testing & Development

**Quick test on a single RAW/TIFF pair**:
```bash
# Place test files in data/
python scripts/fivek_tif_to_xmp.py \
  --root data \
  --split testing \
  --steps 400 \
  --debug_previews
```

**Train on limited data**:
```bash
python run/train_autopilot.py \
  --manifest outputs/dataset_manifest.jsonl \
  --train-limit 100 \
  --val-limit 20 \
  --epochs 5
```

## Device Support

All scripts support MPS (Apple Silicon), CUDA, and CPU:
- Use `--device mps` for M1/M2/M3 Macs
- Use `--device cuda` for NVIDIA GPUs
- Use `--device cpu` as fallback
- Use `--device auto` to auto-select best available device

macOS-specific: Set `KMP_DUPLICATE_LIB_OK=TRUE` if you encounter libomp conflicts (handled in code).

## MIT-Adobe FiveK Dataset

Expected structure:
```
data/MITAboveFiveK/
  raw/**/*.dng                      # RAW files
  processed/tiff16_c/**/*.tif       # Expert C edits (or a,b,d,e)
  training.json                      # Optional: split definitions
  validation.json
  testing.json
```

Use `download_fivek_editor.py` to download the dataset (requires external `dataset/fivek.py` from yuukicammy/mit-adobe-fivek-dataset).

## Important Notes

- **ProPhoto RGB**: All fitting/editing happens in linear ProPhoto. sRGB is only for preview output.
- **Tone curve monotonicity**: Enforced via `softplus(deltas) → cumsum → normalize` to prevent inversions.
- **Parameter clamping**: All denormalized parameters are clamped to legal ACR ranges before export.
- **XMP compatibility**: Output XMPs use ProcessVersion="11.0" (PV2012) and include 256-point tone curves.
- **Training data format**: Model expects `preview` (sRGB PNG) and `data` (JSON with params_norm) fields in manifest.
- **Manifest behavior**:
  - Auto-merges with existing entries on each run (keyed by id, split, expert)
  - Auto-rebuilds from existing output files (scans preview/ and data/ directories)
  - Can safely resume after interruptions or deleted manifest
- **Image-space training**: Use `--use-edit-layer` to optimize for visual quality instead of just parameter accuracy (requires more GPU memory).