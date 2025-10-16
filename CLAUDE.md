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

Or use pip:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies: PyTorch, rawpy, imageio, opencv, tqdm, timm (via pip)

## Quick Reference

**Most common commands:**

1. **Use pretrained model** (recommended for most users):
   ```bash
   # Download checkpoint
   pip install huggingface-hub
   huggingface-cli download Willlllllllllll/Auto_ACR_base Auto_ACR_base_v0.2.pt --local-dir run

   # Render edited TIFF + JPEG from RAW
   python run/predict_and_render.py \
     --checkpoint run/Auto_ACR_base_v0.2.pt \
     --input photo.dng \
     --output outputs/rendered \
     --srgb-output
   ```

2. **Fine-tune with LoRA** (customize style):
   ```bash
   python run/lora_finetune.py \
     --manifest outputs/dataset_manifest.jsonl \
     --checkpoint run/Auto_ACR_base_v0.2.pt \
     --output outputs/lora \
     --epochs 10
   ```

3. **Train from scratch** (advanced):
   ```bash
   # Step 1: Fit parameters from RAW/TIFF pairs
   python scripts/fivek_tif_to_xmp.py \
     --root data/MITAboveFiveK \
     --split training \
     --expert c

   # Step 2: Train model
   python run/train_autopilot.py \
     --manifest outputs/dataset_manifest.jsonl \
     --use-edit-layer \
     --epochs 20
   ```

## Core Workflows

### 1. Inference (Using Pretrained Models)

**Download pretrained checkpoint:**
```bash
# Install huggingface-cli first: pip install huggingface-hub
huggingface-cli download Willlllllllllll/Auto_ACR_base Auto_ACR_base_v0.2.pt --local-dir run
```

**Predict parameters only** (`run/inference.py`):
```bash
python run/inference.py \
  --checkpoint run/Auto_ACR_base_v0.2.pt \
  --input path/to/image.dng \
  --output outputs/predictions \
  --export-json \
  --export-preview
```

**Predict and render edited TIFF** (`run/predict_and_render.py`):
```bash
python run/predict_and_render.py \
  --checkpoint run/Auto_ACR_base_v0.2.pt \
  --input path/to/image.dng \
  --output outputs/rendered \
  --srgb-output
```

**Key inference parameters**:
- `--batch`: Process entire directory of RAW files
- `--export-xmp`: Export XMP sidecar files (default: enabled)
- `--export-json`: Export JSON parameter files
- `--export-preview`: Export sRGB preview with predicted edits
- `--srgb-output`: Save JPEG preview alongside TIFF
- `--lora-checkpoint`: Apply LoRA adapter for style customization
- `--icc-profile`: Path to linear ProPhoto ICC profile (default: `ProPhoto.icm`)

**Outputs**:
- XMP sidecars compatible with Adobe Camera Raw/Lightroom
- TIFF files with full EXIF/metadata preservation
- Optional JPEG previews in sRGB

### 2. LoRA Fine-Tuning

**Script**: `run/lora_finetune.py`

Fine-tune the base model with low-rank adapters for style customization:

```bash
python run/lora_finetune.py \
  --manifest outputs/dataset_manifest.jsonl \
  --checkpoint run/Auto_ACR_base_v0.2.pt \
  --output outputs/lora \
  --epochs 10 \
  --batch-size 8 \
  --lora-rank 12 \
  --train-heads
```

**Key LoRA parameters**:
- `--lora-rank`: Adapter rank (default: 8, higher = more capacity)
- `--lora-alpha`: Scaling factor (default: 16.0)
- `--train-heads`: Also fine-tune slider/tone curve prediction heads
- `--lora-target`: Module patterns to adapt (defaults to backbone attention/MLP layers)
- `--save-frequency`: Save intermediate checkpoints every N epochs

**Outputs**:
- `lora_epochXXX.pt`: LoRA adapter checkpoints (only adapter weights, ~1-5MB)
- Apply with `--lora-checkpoint` flag in inference/rendering scripts

**Published LoRA adapters** available at `Willlllllllllll/Auto_ACR_base` (in `lora/` subfolder)

### 3. Data Preparation (RAW → Parameters)

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

### 4. Model Training (From Scratch)

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

### LoRA Architecture (`model/lora.py`)

**LoRALinear** wraps existing `nn.Linear` layers with low-rank residual updates:

```python
output = base_linear(x) + (alpha / rank) * (x @ lora_down @ lora_up)
```

- `lora_down`: [in_features, rank] initialized with Kaiming uniform
- `lora_up`: [rank, out_features] initialized to zeros (identity at start)
- Base weights are frozen during fine-tuning

**inject_lora()** automatically replaces target layers:
- Default targets: backbone attention (qkv, proj) and MLP (fc1, fc2) layers
- Supports fnmatch patterns for flexible targeting
- Returns list of replaced module names

**LoRA checkpoint format**: Contains only adapter weights (~1-5MB vs ~80MB full model)

### Differentiable Edit Layer (`model/edit_layer.py`)

**DifferentiableEditLayer** applies ACR-style adjustments in a fully differentiable manner:

**Edit pipeline** (applied sequentially in linear ProPhoto RGB):
1. White balance: Channel gains derived from Temperature/Tint
2. Exposure: Power-of-2 multiplication (EV stops)
3. Contrast: Tanh S-curve around 0.5 midpoint
4. Regional adjustments (Highlights, Shadows, Whites, Blacks):
   - Luminance-based masks via sigmoid centered at pivot points
   - Masks are mean-centered to preserve overall exposure
5. Tone curve: Piecewise-linear interpolation with 1024-point LUT
6. Vibrance: Adaptive saturation boost (stronger on desaturated pixels)
7. Saturation: Uniform chroma scaling in YUV-like space

**Key design choices**:
- All operations preserve gradients for backpropagation
- Clamping applied at each stage to prevent numerical instability
- White balance normalized to maintain relative brightness
- Tone curve uses gather-based interpolation for efficiency

**Training usage**: Enable with `--use-edit-layer` flag to optimize predicted parameters for visual quality (image-space loss) instead of just parameter accuracy.

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

**Download the dataset:**
```bash
python download_fivek_editor.py \
  --root data/MITAboveFiveK \
  --splits train val \
  --experts c
```

Requires `dataset/fivek.py` from yuukicammy/mit-adobe-fivek-dataset. Use `--dataset-repo` to specify path if cloned elsewhere.

## Utility Scripts

- `scripts/dng_tif_to_json.py`: Fit parameters for any RAW/TIFF pair outside the FiveK dataset structure
- `scripts/convert_tiff_to_prophoto.py`: Convert gamma-encoded TIFFs to linear ProPhoto TIFFs
- `scripts/split_manifest.py`: Split manifest by expert or other criteria
- `scripts/fix_manifest_paths.py`: Update manifest paths after moving files

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
- **EXIF preservation**: All inference/rendering scripts preserve source EXIF metadata in outputs (TIFF and JPEG).
- **LoRA adapters**: Lightweight (~1-5MB) style customization without retraining full model. Multiple adapters can be created and swapped at inference time.