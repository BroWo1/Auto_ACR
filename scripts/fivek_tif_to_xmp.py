#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert MIT-Adobe FiveK RAW+TIF pairs into Camera Raw XMP sidecars by fitting global PV2012 sliders.
- Supports relative repo paths
- Supports MPS (Metal on Apple), CUDA, or CPU
- Uses JSON splits from yuukicammy/mit-adobe-fivek-dataset if available, else basename matching

Deps:
  pip install torch rawpy imageio opencv-python numpy tqdm

Repo layout (recommended):
acr-autopilot/
  data/MITAboveFiveK/
    raw/**/<name>.dng
    processed/tiff16_c/**/<name>.tif
    training.json validation.json testing.json  # optional, from the helper repo
  outputs/xmps/
  outputs/1_preview/
  outputs/1_data/
  scripts/fivek_tif_to_xmp.py
"""

import os, sys, json, math, argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import imageio.v3 as iio
import rawpy
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import io
import json as _json

try:
    from PIL import Image, ImageCms
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# --- macOS stock ICC profiles (present on your machine) ---
SRGB_ICC_PATH = "/System/Library/ColorSync/Profiles/sRGB Profile.icc"
# Apple ships ROMM RGB.icc (ProPhoto, gamma 1.8); there is no linear ROMM by default
ROMM_ICC_PATH = "/System/Library/ColorSync/Profiles/ROMM RGB.icc"

# -------------------------------
# Camera Raw PV2012 slider ranges (per Adobe crs schema)
# -------------------------------

CRS_RANGES = {
    "Temperature": (2000.0, 50000.0),
    "Tint": (-150.0, 150.0),
    "Exposure2012": (-5.0, 5.0),
    "Contrast2012": (-100.0, 100.0),
    "Highlights2012": (-100.0, 100.0),
    "Shadows2012": (-100.0, 100.0),
    "Whites2012": (-100.0, 100.0),
    "Blacks2012": (-100.0, 100.0),
    "Vibrance": (-100.0, 100.0),
    "Saturation": (-100.0, 100.0),
}

# -------------------------------
# Color utilities (ProPhoto -> sRGB preview)
# -------------------------------

def _srgb_encode(linear: torch.Tensor) -> torch.Tensor:
    a = 0.055
    thresh = 0.0031308
    return torch.where(
        linear <= thresh,
        12.92 * linear,
        (1 + a) * torch.pow(linear.clamp(min=0), 1/2.4) - a,
    )

def _mat3_mul(img: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    img: [..., 3, H, W] or [1,3,H,W] or [3,H,W]; applies 3x3 matrix M on channel axis.
    Returns same shape as input.
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    B, C, H, W = img.shape
    x = img.reshape(B, C, -1)  # [B,3,N]
    y = torch.matmul(M.to(img.device).unsqueeze(0).expand(B, -1, -1), x)  # [B,3,N]
    y = y.reshape(B, C, H, W)
    return y.squeeze(0) if squeeze else y

def _prophoto_to_srgb_matrix() -> torch.Tensor:
    """
    Build a 3x3 linear ProPhoto(D50) -> linear sRGB(D65) matrix using Bradford CAT.
    Constants from published definitions of ROMM/ProPhoto, sRGB, and Bradford.
    """
    # ProPhoto RGB (ROMM) to XYZ (D50)
    P2XYZ = torch.tensor([
        [0.7976749, 0.1351917, 0.0313534],
        [0.2880402, 0.7118741, 0.0000857],
        [0.0000000, 0.0000000, 0.8252100],
    ], dtype=torch.float32)

    # sRGB <-> XYZ (D65)
    XYZ2sRGB = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=torch.float32)

    # Bradford matrices
    B = torch.tensor([
        [ 0.8951000,  0.2664000, -0.1614000],
        [-0.7502000,  1.7135000,  0.0367000],
        [ 0.0389000, -0.0685000,  1.0296000],
    ], dtype=torch.float32)
    B_inv = torch.tensor([
        [ 0.9869929, -0.1470543,  0.1599627],
        [ 0.4323053,  0.5183603,  0.0492912],
        [-0.0085287,  0.0400428,  0.9684867],
    ], dtype=torch.float32)

    # White points
    D50 = torch.tensor([0.96422, 1.00000, 0.82521], dtype=torch.float32)
    D65 = torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32)

    # Bradford CAT D50 -> D65
    cone_src = B @ D50
    cone_dst = B @ D65
    M_adapt = B_inv @ torch.diag(cone_dst / cone_src) @ B

    # ProPhoto(D50) -> XYZ(D65)
    P2XYZ_D65 = M_adapt @ P2XYZ
    # ProPhoto(D50, linear) -> sRGB(D65, linear)
    P2sRGB = XYZ2sRGB @ P2XYZ_D65
    return P2sRGB

_P2SRGB = _prophoto_to_srgb_matrix()

def _prophoto_encode_torch(linear: torch.Tensor) -> torch.Tensor:
    """
    Encode linear ProPhoto (ROMM) to gamma-encoded ROMM in [0,1].
    Piecewise with small linear segment near black.
    """
    thresh = 1.0 / 512.0  # threshold in linear domain
    return torch.where(
        linear <= thresh,
        linear * 16.0,
        torch.clamp(linear, 0, 1) ** (1/1.8)
    )

def _prophoto_decode_torch(encoded: torch.Tensor) -> torch.Tensor:
    """
    Decode gamma-encoded ROMM/ProPhoto to linear.
    Inverse of _prophoto_encode_torch.
    """
    enc_thresh = 1.0 / 32.0  # 16 * (1/512)
    return torch.where(
        encoded <= enc_thresh,
        encoded / 16.0,
        torch.clamp(encoded, 0, 1) ** 1.8
    )

def _to_pil_rgb8(arr01: np.ndarray):
    if not HAS_PIL:
        return None
    arr8 = np.clip(arr01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr8, mode="RGB")

def _from_pil_rgb8(img) -> np.ndarray:
    if not HAS_PIL:
        return None
    arr = np.asarray(img, dtype=np.uint8)
    return (arr.astype(np.float32) / 255.0)

def _icc_transform_rgb(img_rgb01: np.ndarray, src_icc, intent=None, bpc=True) -> np.ndarray:
    if not HAS_PIL:
        return img_rgb01
    if intent is None:
        intent = ImageCms.INTENT_RELATIVE_COLORIMETRIC
    pil = _to_pil_rgb8(img_rgb01)
    dst_icc = ImageCms.createProfile("sRGB")
    xfm = ImageCms.buildTransform(src_icc, dst_icc, "RGB", "RGB",
                                  renderingIntent=intent, blackPointCompensation=bpc)
    out = ImageCms.applyTransform(pil, xfm)
    return _from_pil_rgb8(out)

def prophoto_to_srgb_preview(linear_prophoto: torch.Tensor,
                             prophoto_icc_path: Optional[str] = None) -> np.ndarray:
    """
    Convert **linear** ProPhoto (ROMM, D50) [1,3,H,W] to sRGB [0,1] using ICC (RelCol+BPC).
    On macOS, uses Apple ROMM RGB.icc and sRGB Profile.icc. Falls back to matrix+Bradford.
    """
    # 1) encode linear ProPhoto -> gamma 1.8 ROMM values (what the ROMM ICC expects)
    with torch.no_grad():
        enc_romm = _prophoto_encode_torch(linear_prophoto).clamp(0,1)
        enc_np = enc_romm.squeeze(0).permute(1,2,0).cpu().numpy()

    if HAS_PIL:
        # Prefer user-specified ICC, else system ROMM RGB.icc
        src_icc = None
        try:
            if prophoto_icc_path and Path(prophoto_icc_path).exists():
                src_icc = ImageCms.getOpenProfile(prophoto_icc_path)
        except Exception:
            src_icc = None
        if src_icc is None:
            try:
                src_icc = ImageCms.getOpenProfile(ROMM_ICC_PATH)
            except Exception:
                src_icc = None
        try:
            dst_icc = ImageCms.getOpenProfile(SRGB_ICC_PATH)
        except Exception:
            dst_icc = ImageCms.createProfile("sRGB")

        if src_icc is not None:
            try:
                pil = _to_pil_rgb8(enc_np)
                xfm = ImageCms.buildTransform(src_icc, dst_icc, "RGB", "RGB",
                                              renderingIntent=ImageCms.INTENT_RELATIVE_COLORIMETRIC,
                                              blackPointCompensation=True)
                out = ImageCms.applyTransform(pil, xfm)
                return _from_pil_rgb8(out)
            except Exception:
                pass

    # 2) fallback: matrix + Bradford (approximate)
    srgb_lin = _mat3_mul(linear_prophoto, _P2SRGB).clamp(0,1)
    return _to_numpy_img01(_srgb_encode(srgb_lin))

# -------------------------------
# Normalize / denormalize helpers
# -------------------------------

def denorm_param(name: str, y: float) -> float:
    """Map normalized [-1,1] -> CRS range, with hard clamp on input."""
    lo, hi = CRS_RANGES[name]
    y = max(-1.0, min(1.0, float(y)))
    return float((y + 1.0) * 0.5 * (hi - lo) + lo)

def denorm_dict(params_norm: Dict[str, np.ndarray]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in params_norm.items():
        if k == "ToneCurve":
            out[k] = np.asarray(v, dtype=np.float32)  # keep knots in [0,1]
        else:
            out[k] = denorm_param(k, float(v))
    return out

# Torch-safe denorm (keeps gradients if used inside graph)
def denorm_param_torch(name: str, y: torch.Tensor) -> torch.Tensor:
    """Torch version with clamp so training stays within ACR-legal bounds."""
    lo, hi = CRS_RANGES[name]
    y = y.clamp(-1.0, 1.0)
    return (y + 1.0) * 0.5 * (hi - lo) + lo

# -------------------------------
# RAW/TIF I/O in (linear-ish) ProPhoto
# -------------------------------

def read_raw_linear_prophoto(dng_path: Path, long_side=512) -> torch.Tensor:
    """
    Decode RAW to a 16-bit ProPhoto RGB image with gamma=(1,1) (linear-ish),
    then resize to a 512px-long-side thumbnail and return [1,3,H,W] float in [0,1].
    """
    with rawpy.imread(str(dng_path)) as r:
        # no_auto_bright avoids hidden exposure changes; gamma=(1,1) approximates linear;
        # ColorSpace.ProPhoto aligns with FiveK target TIFF space.
        rgb16 = r.postprocess(
            output_bps=16,
            no_auto_bright=True,
            gamma=(1, 1),
            output_color=rawpy.ColorSpace.ProPhoto,
            use_camera_wb=True,
            # Let libraw apply the as-shot orientation so RAW matches processed TIFF
            # (set user_flip=None by omitting it)
        )
    h, w = rgb16.shape[:2]
    s = float(long_side) / max(h, w)
    if s < 1.0:
        rgb16 = cv2.resize(rgb16, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb16).permute(2, 0, 1).float().unsqueeze(0) / 65535.0
    return x  # [1,3,H,W], ProPhoto, ~linear, [0,1]

def read_tif16(path: Path, long_side: Optional[int] = None) -> torch.Tensor:
    """
    Read FiveK processed TIFF (typically **gamma ProPhoto / ROMM**) and return
    **linear ProPhoto** tensor [1,3,H,W] in [0,1] for fitting.
    If ICC present and recognized, we assume ROMM gamma and decode; otherwise we
    conservatively decode as ROMM gamma as well (FiveK targets are in ProPhoto).
    """
    # 1) read pixels (fast) via imageio
    arr = iio.imread(str(path))  # HxWxC, dtype uint16/float
    if long_side is not None:
        h, w = arr.shape[:2]
        s = float(long_side) / max(h, w)
        if s < 1.0:
            arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    # Normalize to [0,1]
    if np.issubdtype(arr.dtype, np.floating):
        arr01 = np.clip(arr, 0.0, 1.0).astype(np.float32)
    elif np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr01 = (arr.astype(np.float32) / float(info.max))
    else:
        arr01 = arr.astype(np.float32)
        arr01 = np.clip(arr01 / (arr01.max() if arr01.max() > 0 else 1.0), 0.0, 1.0)

    # 2) detect ICC (slow path) with Pillow, if available
    assume_romm_gamma = True
    if HAS_PIL:
        try:
            pil = Image.open(str(path))
            icc_bytes = pil.info.get("icc_profile", None)
            if icc_bytes:
                # If it's ROMM/ProPhoto, keep assume_romm_gamma=True; otherwise leave as-is
                tag = icc_bytes[:128]  # quick sniff
                if (b"ROMM" in tag) or (b"ProPhoto" in tag):
                    assume_romm_gamma = True
        except Exception:
            pass

    # 3) If gamma-encoded ROMM, decode -> linear for fitting
    t = torch.from_numpy(arr01).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    if assume_romm_gamma:
        y_lin = _prophoto_decode_torch(t)
    else:
        y_lin = t
    return y_lin  # linear ProPhoto [1,3,H,W]

def read_tif16_color_managed_to_srgb_png(path: Path, long_side: Optional[int] = None) -> np.ndarray:
    """
    Read TIFF with its embedded ICC (if any) and convert to sRGB via LittleCMS
    (Relative Colorimetric + BPC). Returns HxWx3 float [0,1] in sRGB (gamma-encoded).
    Falls back to matrix path if Pillow/LittleCMS not available.
    """
    if not HAS_PIL:
        # Fallback: approximate via matrix path using our linear reader
        y_pp = read_tif16(path, long_side=long_side)
        srgb_lin = _mat3_mul(y_pp, _P2SRGB).clamp(0,1)
        return _to_numpy_img01(_srgb_encode(srgb_lin))

    pil = Image.open(str(path))
    # resize early to minimize color roundoff
    if long_side is not None:
        w, h = pil.size
        s = float(long_side) / max(h, w)
        if s < 1.0:
            pil = pil.resize((int(w*s), int(h*s)), resample=Image.Resampling.LANCZOS)

    icc_bytes = pil.info.get("icc_profile", None)
    src_icc = ImageCms.getOpenProfile(io.BytesIO(icc_bytes)) if icc_bytes else None
    dst_icc = ImageCms.createProfile("sRGB")
    if src_icc:
        xfm = ImageCms.buildTransform(src_icc, dst_icc, pil.mode, "RGB",
                                      renderingIntent=ImageCms.INTENT_RELATIVE_COLORIMETRIC,
                                      blackPointCompensation=True)
        out = ImageCms.applyTransform(pil.convert("RGB"), xfm)
        return _from_pil_rgb8(out)
    else:
        # Assume ROMM/ProPhoto gamma-encoded input; decode → linear → matrix → sRGB encode
        np01 = np.asarray(pil.convert("RGB"), dtype=np.uint8).astype(np.float32)/255.0
        enc = torch.from_numpy(np01).permute(2,0,1).unsqueeze(0)
        lin = _prophoto_decode_torch(enc)
        srgb = _srgb_encode(_mat3_mul(lin, _P2SRGB)).clamp(0,1)
        return _to_numpy_img01(srgb)

# -------------------------------
# Differentiable "mini-ACR" edit layer
# -------------------------------

def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def apply_wb(x, temp, tint):
    """
    Approximate WB via channel gains derived from normalized temp/tint.
    This is heuristic; exact mapping is camera/profile dependent.
    """
    dT  = temp * 0.5
    dTi = tint * 0.2
    g_r = torch.exp( 0.7 * dT - 0.8 * dTi)
    g_b = torch.exp(-0.7 * dT + 0.8 * dTi)
    gains = torch.stack([g_r, torch.ones_like(g_r), g_b], dim=1).unsqueeze(-1).unsqueeze(-1)
    return x * gains

def apply_exposure(x, ev_real):
    """
    Exposure in EV (stops). +1 doubles, -1 halves.
    """
    gain = (2.0 ** ev_real).view(-1, 1, 1, 1)
    return x * gain

def apply_wb_kelvin(x: torch.Tensor, tempK: torch.Tensor, tint: torch.Tensor):
    """
    Approx WB gains from Kelvin/Tint, anchored at ~D65.
    Gentle slopes so optimizer stays stable and matches XMP units.
    """
    # normalize deltas
    dT  = (tempK.clamp(2000.0, 50000.0) - 6500.0) / 6500.0     # around 0 at D65
    dTi = tint.clamp(-150.0, 150.0) / 150.0                    # [-1..1]

    # tempered coefficients
    kr, kb = 0.20, 0.20
    mr, mb = 0.30, 0.30
    g_r = torch.exp( kr * dT - mr * dTi )
    g_b = torch.exp( -kb * dT + mb * dTi )
    gains = torch.stack([g_r, torch.ones_like(g_r), g_b], dim=1).unsqueeze(-1).unsqueeze(-1)
    return (x * gains).clamp(0, 1)

def apply_contrast(x, contrast):
    """
    Contrast control that is identity at 0 and smoothly increases/decreases
    contrast as contrast -> +/-1 by blending a tanh S-curve with identity.
    """
    c = contrast.clamp(-1, 1).view(-1, 1, 1, 1)
    y = x - 0.5
    k = 1.5  # base steepness for the nonlinear curve
    # Use tensor denominator to avoid calling torch.tanh on a Python float
    denom = torch.tanh(torch.tensor(2.0 * k, device=x.device, dtype=x.dtype)) + 1e-6
    y_nl = torch.tanh(k * 2.0 * y) / denom
    y2 = y + c * (y_nl - y)  # when c=0 -> identity
    return (y2 + 0.5).clamp(0, 1)

def apply_region_adjustments(x, high, shad, whites, blacks):
    y = rgb_to_luma(x)
    k = 12.0
    m_high  = torch.sigmoid((y - 0.6) * k)
    m_shad  = 1.0 - torch.sigmoid((y - 0.4) * k)
    m_white = torch.sigmoid((y - 0.9) * k)
    m_black = 1.0 - torch.sigmoid((y - 0.1) * k)

    def ofs(z, scale):
        return (z.view(-1,1,1,1) * scale)

    y2 = (y
          + ofs(high, 0.15)  * m_high
          + ofs(shad, 0.15)  * m_shad
          + ofs(whites, 0.20)* m_white
          + ofs(blacks, 0.20)* m_black).clamp(0, 1)

    ratio = (y2 + 1e-6) / (y + 1e-6)
    return (x * ratio).clamp(0,1)

def tonecurve_from_knots(knots, steps=256):
    """
    Build a [B,1,1,steps] LUT via piecewise-linear interpolation.
    The provided knots are interior shape controls in [0,1]; we add explicit
    endpoints 0 and 1 so that a uniformly spaced interior (the default) yields
    the identity mapping.
    """
    B, K = knots.shape  # knots: [B,K] in [0,1]
    device = knots.device
    # Add endpoints
    yext = torch.cat([torch.zeros(B, 1, device=device), knots.clamp(0,1), torch.ones(B, 1, device=device)], dim=1)
    Kext = K + 2
    # Sample positions in [0,1]
    xs = torch.linspace(0.0, 1.0, steps=steps, device=device).unsqueeze(0).expand(B, steps)  # [B,S]
    # Map sample positions to segment indices in [0, Kext-2]
    pos = xs * (Kext - 1)
    i0 = pos.floor().to(torch.long).clamp(0, Kext - 2)  # [B,S]
    t = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)        # [B,S]
    # Gather neighboring knot values
    y0 = torch.gather(yext, 1, i0)                      # [B,S]
    y1 = torch.gather(yext, 1, i0 + 1)                  # [B,S]
    y = y0 * (1.0 - t) + y1 * t                         # [B,S]
    return y.reshape(B, 1, 1, steps).clamp(0, 1)

def apply_tone_curve(x, knots):
    """
    Apply per-batch monotone tone curve defined by K knots directly via piecewise-linear
    interpolation without a 256-entry LUT. This avoids large temporary tensors and works
    across devices.
    x: [B,C,H,W] in [0,1]
    knots: [B,K] in [0,1]
    """
    B, C, H, W = x.shape
    Bk, K = knots.shape
    assert B == Bk, "Batch size mismatch between image and tone-curve knots"

    pos = (x.clamp(0, 1) * (K - 1))
    i0 = pos.floor().to(torch.long).clamp(0, K - 2)        # [B,C,H,W]
    t = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)           # [B,C,H,W]

    # Broadcast knots to [B,K,C,H,W] and gather y0/y1 along knot-dimension
    knots_b = knots.reshape(B, K, 1, 1, 1).expand(B, K, C, H, W)
    i0b = i0.unsqueeze(1)                                  # [B,1,C,H,W]
    y0 = torch.gather(knots_b, 1, i0b).squeeze(1)          # [B,C,H,W]
    y1 = torch.gather(knots_b, 1, (i0b + 1).clamp(max=K-1)).squeeze(1)  # [B,C,H,W]

    y = y0 * (1.0 - t) + y1 * t
    return y.clamp(0, 1)

def rgb_to_yuv(x):
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.14713*r - 0.28886*g + 0.436*b
    v =  0.61500*r - 0.51499*g - 0.10001*b
    return torch.cat([y,u,v], dim=1)

def yuv_to_rgb(yuv):
    y, u, v = yuv[:,0:1], yuv[:,1:2], yuv[:,2:3]
    r = y + 1.13983*v
    g = y - 0.39465*u - 0.58060*v
    b = y + 2.03211*u
    return torch.cat([r,g,b], dim=1)

def apply_vibrance_saturation(x, vibrance, saturation):
    yuv = rgb_to_yuv(x)
    y, u, v = yuv[:,0:1], yuv[:,1:2], yuv[:,2:3]
    sat_mag = torch.sqrt(u*u + v*v + 1e-6)
    vib = 1.0 + (vibrance.view(-1,1,1,1) * 0.6)
    vib_gain = 1.0 + (1.0 - torch.tanh(3.0*sat_mag)) * (vib - 1.0)
    sat_gain = 1.0 + (saturation.view(-1,1,1,1) * 0.01)
    gain = vib_gain * sat_gain
    u2, v2 = u * gain, v * gain
    return yuv_to_rgb(torch.cat([y,u2,v2], dim=1)).clamp(0,1)


# -------------------------------
# Portable “real-units” edit (no Theta) for baking LUTs / alt. formats
# -------------------------------

def edit_layer_realunits(x: torch.Tensor, params: Dict[str, object]) -> torch.Tensor:
    """Apply fitted edit using real ACR-like units from params_denorm."""

    x = x.clamp(0, 1)

    def to_tensor(val: object) -> torch.Tensor:
        return torch.tensor(float(val), dtype=x.dtype, device=x.device).view(1)

    tempK = to_tensor(params["Temperature"])   # Kelvin (2000..50000)
    tint = to_tensor(params["Tint"])           # -150..150
    ev = to_tensor(params["Exposure2012"])     # -5..5 stops
    con = to_tensor(params["Contrast2012"])    # -100..100
    hi = to_tensor(params["Highlights2012"])   # -100..100
    sh = to_tensor(params["Shadows2012"])      # -100..100
    wh = to_tensor(params["Whites2012"])       # -100..100
    bl = to_tensor(params["Blacks2012"])       # -100..100
    vib = to_tensor(params["Vibrance"])        # -100..100
    sat = to_tensor(params["Saturation"])      # -100..100

    # WB + Exposure
    x = apply_wb_kelvin(x, tempK, tint)
    x = apply_exposure(x, ev)

    # Normalize remaining sliders to [-1,1]-ish
    c = (con / 100.0).view(1)
    h = (hi / 100.0).view(1)
    s = (sh / 100.0).view(1)
    w = (wh / 100.0).view(1)
    b = (bl / 100.0).view(1)
    vbn = (vib / 100.0).view(1)
    stn = (sat / 100.0).view(1)

    x = apply_contrast(x, c)
    x = apply_region_adjustments(x, h, s, w, b)

    knots = torch.tensor(np.asarray(params["ToneCurve"], dtype=np.float32),
                         dtype=x.dtype, device=x.device).view(1, -1)
    x = apply_tone_curve(x, knots)
    x = apply_vibrance_saturation(x, vbn, stn)
    return x.clamp(0, 1)


# -------------------------------
# Exports: JSON + 3D LUT (.cube)
# -------------------------------

def export_params_json(params_denorm: Dict[str, object],
                       params_norm: Dict[str, float],
                       out_path: Path) -> None:
    """Write portable JSON snapshot of fitted parameters."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    out_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")


def export_lut3d_cube_prophoto_linear(params_denorm: Dict[str, object],
                                      size: int,
                                      out_path: Path,
                                      batch: int = 16384) -> None:
    """Bake a .cube LUT in linear ProPhoto RGB domain."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = torch.linspace(0, 1, size)
    rr, gg, bb = torch.meshgrid(grid, grid, grid, indexing="ij")
    rgb = torch.stack([rr, gg, bb], dim=-1).reshape(-1, 3)

    device = torch.device("cpu")
    ys = []
    for start in range(0, rgb.shape[0], batch):
        chunk = rgb[start:start + batch].t().unsqueeze(0).unsqueeze(2)  # [1,3,1,w]
        y = edit_layer_realunits(chunk.to(device), params_denorm)
        ys.append(y.squeeze().t().cpu().numpy())  # -> [w,3]

    lut = np.concatenate(ys, axis=0)
    lut = np.clip(lut, 0.0, 1.0)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# domain: ProPhoto RGB (linear), range [0..1]\n")
        f.write("TITLE \"acr-autopilot baked look\"\n")
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
        for r, g, b in lut:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")

class Theta(torch.nn.Module):
    """
    Normalized sliders in [-1,1] and a monotone tone-curve parameterized via positive deltas.
    """
    def __init__(self, K=8, device="cpu"):
        super().__init__()
        self.temp = torch.nn.Parameter(torch.zeros(1, device=device))
        self.tint = torch.nn.Parameter(torch.zeros(1, device=device))
        self.ev   = torch.nn.Parameter(torch.zeros(1, device=device))
        self.contrast = torch.nn.Parameter(torch.zeros(1, device=device))
        self.high = torch.nn.Parameter(torch.zeros(1, device=device))
        self.shad = torch.nn.Parameter(torch.zeros(1, device=device))
        self.whi  = torch.nn.Parameter(torch.zeros(1, device=device))
        self.bla  = torch.nn.Parameter(torch.zeros(1, device=device))
        self.vib  = torch.nn.Parameter(torch.zeros(1, device=device))
        self.sat  = torch.nn.Parameter(torch.zeros(1, device=device))
        self.curve_deltas = torch.nn.Parameter(torch.zeros(K, device=device))

    def curve_knots(self):
        d = F.softplus(self.curve_deltas) + 1e-4  # positive
        y = torch.cumsum(d, dim=0)
        y = y / y[-1].clamp(min=1e-6)
        return y.reshape(1,-1)  # [1,K]

def edit_layer(x, theta: Theta):
    x = x.clamp(0, 1)

    # denormalize to real units AS TENSORS (keep gradients!)
    tempK = denorm_param_torch("Temperature",   theta.temp)       # Kelvin
    tint  = denorm_param_torch("Tint",          theta.tint)       # -150..150
    ev    = denorm_param_torch("Exposure2012",  theta.ev)         # EV stops
    con   = denorm_param_torch("Contrast2012",  theta.contrast)   # -100..100
    hi    = denorm_param_torch("Highlights2012",theta.high)       # -100..100
    sh    = denorm_param_torch("Shadows2012",   theta.shad)       # -100..100
    wh    = denorm_param_torch("Whites2012",    theta.whi)        # -100..100
    bl    = denorm_param_torch("Blacks2012",    theta.bla)        # -100..100
    vib   = denorm_param_torch("Vibrance",      theta.vib)        # -100..100
    sat   = denorm_param_torch("Saturation",    theta.sat)        # -100..100

    # WB + exposure in real units
    x = apply_wb_kelvin(x, tempK.view(1), tint.view(1))
    x = apply_exposure(x, ev.view(1))

    # downstream ops expect roughly [-1..1] amplitudes
    c   = (con / 100.0).view(1)
    h   = (hi  / 100.0).view(1)
    s   = (sh  / 100.0).view(1)
    w   = (wh  / 100.0).view(1)
    b   = (bl  / 100.0).view(1)
    vbn = (vib / 100.0).view(1)
    stn = (sat / 100.0).view(1)

    x = apply_contrast(x, c)
    x = apply_region_adjustments(x, h, s, w, b)
    x = apply_tone_curve(x, theta.curve_knots())
    x = apply_vibrance_saturation(x, vbn, stn)
    return x.clamp(0, 1)

# -------------------------------
# Losses & regularizers
# -------------------------------

def image_loss(y_hat, y):
    l1 = (y_hat - y).abs().mean()
    # simple SSIM-like proxy (kept lightweight and MPS-friendly)
    num = ((y_hat - y)**2).mean()
    den = y_hat.pow(2).mean() + y.pow(2).mean() + 1e-6
    ssim_proxy = 1.0 - (1.0 - (num / den)).clamp(0,1)
    return l1 + 0.2*ssim_proxy

def curve_reg(theta: Theta):
    d = F.softplus(theta.curve_deltas)
    smooth = (d[1:] - d[:-1]).pow(2).mean()
    return 0.01*smooth

# -------------------------------
# XMP writing (crs PV2012)
# -------------------------------

def tonecurve_knots_to_pairs(knots: np.ndarray) -> List[str]:
    """
    Resample learned knots to a 256-point PV2012 curve.
    - X runs 0..255 strictly increasing
    - Y is clamped to 0..255 and forced nondecreasing
    """
    k = np.asarray(knots, dtype=np.float32).clip(0, 1)
    K = len(k)
    xs = np.arange(256, dtype=np.int32)  # 0..255
    # piecewise-linear interp from knots -> 256 samples
    pos = xs * ((K - 1) / 255.0)
    i0 = np.floor(pos).astype(np.int32)
    i0 = np.clip(i0, 0, K - 2)
    t = (pos - i0).astype(np.float32)
    y = (k[i0] * (1.0 - t) + k[i0 + 1] * t)

    y = np.clip(np.round(y * 255.0), 0, 255).astype(np.int32)
    # enforce monotone Y and exact endpoints
    y = np.maximum.accumulate(y)
    y[0] = 0
    y[-1] = 255
    return [f"{int(x)}, {int(yy)}" for x, yy in zip(xs, y)]

def _fmt_signed(v: float, decimals=0, force_plus=True) -> str:
    s = f"{v:.{decimals}f}"
    return f"+{s}" if force_plus and not s.startswith("-") else s

def _is_near_linear_curve(knots: np.ndarray, tol: float = 1e-3) -> bool:
    x = np.linspace(0.0, 1.0, len(knots), dtype=np.float32)
    k = np.asarray(knots, dtype=np.float32)
    return np.max(np.abs(k - x)) <= tol

def xmp_from_params(params: Dict[str, object],
                    camera_profile: str = "Adobe Color",
                    process_version: Optional[str] = "11.0",
                    version: Optional[str] = None) -> str:
    """
    Build a PV2012-style XMP.
    - Write a valid ToneCurvePV2012 with strictly increasing X and endpoints.
    - Set ToneCurveName2012 to "Custom" if the curve isn't linear.
    - Make ProcessVersion/Version optional (ACR/LR will fill if omitted).
    """
    tc_pairs = tonecurve_knots_to_pairs(params["ToneCurve"])
    tonecurve_name = "Linear" if _is_near_linear_curve(params["ToneCurve"]) else "Custom"

    # Clamp outgoing values to legal ranges one more time (belt & suspenders)
    def _clamp(name):
        lo, hi = CRS_RANGES[name]
        return float(np.clip(params[name], lo, hi))

    attrs = [
        ("crs:Version", version),
        ("crs:ProcessVersion", process_version),
        ("crs:WhiteBalance", "Custom"),
        ("crs:Temperature", f"{int(round(_clamp('Temperature')))}"),
        ("crs:Tint", _fmt_signed(_clamp('Tint'), decimals=0)),
        ("crs:Exposure2012", _fmt_signed(_clamp('Exposure2012'), decimals=2)),
        ("crs:Contrast2012", _fmt_signed(_clamp('Contrast2012'), decimals=0)),
        ("crs:Highlights2012", _fmt_signed(_clamp('Highlights2012'), decimals=0)),
        ("crs:Shadows2012", _fmt_signed(_clamp('Shadows2012'), decimals=0)),
        ("crs:Whites2012", _fmt_signed(_clamp('Whites2012'), decimals=0)),
        ("crs:Blacks2012", _fmt_signed(_clamp('Blacks2012'), decimals=0)),
        ("crs:Vibrance", _fmt_signed(_clamp('Vibrance'), decimals=0)),
        ("crs:Saturation", _fmt_signed(_clamp('Saturation'), decimals=0)),
        ("crs:ToneCurveName2012", tonecurve_name),
        ("crs:CameraProfile", camera_profile),
        ("crs:HasSettings", "True"),
        # Neutralize modules we don't model so ACR doesn't change pixels
        ("crs:Clarity2012", "+0"),
        ("crs:Dehaze", "+0"),
        ("crs:Sharpness", "0"),
        ("crs:LuminanceSmoothing", "0"),
        ("crs:ColorNoiseReduction", "0"),
        ("crs:AutoLateralCA", "0"),
        ("crs:LensProfileEnable", "0"),
    ]
    attr_str = "\n   ".join(f'{k}="{v}"' for k, v in attrs if v is not None)

    return f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   {attr_str}>
   <crs:ToneCurvePV2012>
    <rdf:Seq>
     {''.join(f'<rdf:li>{p}</rdf:li>' for p in tc_pairs)}
    </rdf:Seq>
   </crs:ToneCurvePV2012>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""

# -------------------------------
# Fitting one RAW/TIF pair
# -------------------------------

@torch.no_grad()
def _to_numpy_img01(t: torch.Tensor) -> np.ndarray:
    # [1,3,H,W] -> HxWx3 float [0,1]
    return t.squeeze(0).clamp(0,1).permute(1,2,0).cpu().numpy()

def fit_params_for_pair(raw_path: Path, tif_path: Path, device: torch.device,
                        long_side=512, steps=800, lr=5e-2, log_every: int = 0,
                        curve_knots: int = 8):
    x = read_raw_linear_prophoto(raw_path, long_side=long_side).to(device)  # [1,3,H,W]
    y = read_tif16(tif_path, long_side=long_side).to(device)                # [1,3,H,W]
    # Ensure same spatial size (FiveK processed TIFFs may differ slightly in aspect/crop)
    if x.shape[-2:] != y.shape[-2:]:
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

    theta = Theta(K=curve_knots, device=str(device))
    opt = torch.optim.Adam(theta.parameters(), lr=lr)

    # Initial loss
    with torch.no_grad():
        y_hat0 = edit_layer(x, theta)
        loss0 = image_loss(y_hat0, y) + curve_reg(theta)
    if log_every and steps > 0:
        print(f"[dbg] init loss={loss0.item():.6f}")

    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        y_hat = edit_layer(x, theta)
        loss = image_loss(y_hat, y) + curve_reg(theta)
        loss.backward()
        opt.step()
        if log_every and ((it + 1) % log_every == 0 or it == steps - 1):
            with torch.no_grad():
                vals = {
                    'ev': theta.ev.item(), 'temp': theta.temp.item(), 'tint': theta.tint.item(),
                    'ct': theta.contrast.item(), 'hi': theta.high.item(), 'sh': theta.shad.item(),
                    'wh': theta.whi.item(), 'bl': theta.bla.item(), 'vb': theta.vib.item(), 'sa': theta.sat.item()
                }
                print(f"[dbg] step {it+1}/{steps} loss={loss.item():.6f} params={vals}")

    params_norm = {
        "Temperature": theta.temp.detach().item(),
        "Tint": theta.tint.detach().item(),
        "Exposure2012": theta.ev.detach().item(),
        "Contrast2012": theta.contrast.detach().item(),
        "Highlights2012": theta.high.detach().item(),
        "Shadows2012": theta.shad.detach().item(),
        "Whites2012": theta.whi.detach().item(),
        "Blacks2012": theta.bla.detach().item(),
        "Vibrance": theta.vib.detach().item(),
        "Saturation": theta.sat.detach().item(),
        "ToneCurve": theta.curve_knots().detach().cpu().numpy()[0],
    }
    params_denorm = denorm_dict(params_norm)
    with torch.no_grad():
        preview_lin_pp = edit_layer(x, theta).clamp(0,1)  # linear ProPhoto
    # ICC-managed sRGB preview (Photoshop-like) with fallback; uses system ROMM/sRGB profiles
    preview_srgb01 = prophoto_to_srgb_preview(preview_lin_pp)
    return params_denorm, params_norm, preview_srgb01

# -------------------------------
# Reading FiveK splits (helper repo) or fallback
# -------------------------------

def load_split_paths(root: Path, expert: str, split: str) -> List[Tuple[Path,Path]]:
    """Return (raw_path, tif_path) pairs by scanning RAWs and matching the expert's TIFFs."""

    # Lightweight testing mode: allow a single ad-hoc pair placed anywhere under --root
    if split == "testing":
        raw_exts = ["*.dng"]
        tif_exts = ["*.tif", "*.tiff"]
        raw_list: List[Path] = []
        tif_list: List[Path] = []
        for pat in raw_exts:
            raw_list.extend(root.rglob(pat))
        for pat in tif_exts:
            tif_list.extend(root.rglob(pat))

        if not raw_list or not tif_list:
            return []

        tif_map = {p.stem: p for p in sorted(tif_list)}
        for raw_path in sorted(raw_list):
            tif_path = tif_map.get(raw_path.stem)
            if tif_path is not None:
                return [(raw_path, tif_path)]

        return [(sorted(raw_list)[0], sorted(tif_list)[0])]

    raw_dir = root / "raw"
    tif_dir = root / f"processed/tiff16_{expert}"

    if not raw_dir.exists() or not tif_dir.exists():
        return []

    allowed_stems: Optional[set[str]] = None
    split_json = root / f"{split}.json"
    if split_json.exists():
        raw_entries = json.loads(split_json.read_text())
        allowed_stems = set()

        for entry in raw_entries:
            if isinstance(entry, dict):
                basename = entry.get("basename")
                if basename:
                    allowed_stems.add(Path(basename).stem)
                continue

            if isinstance(entry, str):
                allowed_stems.add(Path(entry).stem)

        if not allowed_stems:
            allowed_stems = None

    def _find_tif(stem: str) -> Optional[Path]:
        for ext in (".tif", ".tiff"):
            match = next(tif_dir.glob(f"**/{stem}{ext}"), None)
            if match is not None:
                return match
        return None

    pairs: List[Tuple[Path, Path]] = []
    for raw_path in sorted(raw_dir.rglob("*.dng")):
        stem = raw_path.stem
        if allowed_stems is not None and stem not in allowed_stems:
            continue
        tif_path = _find_tif(stem)
        if tif_path is None:
            continue
        pairs.append((raw_path, tif_path))
    return pairs

# -------------------------------
# Main
# -------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "data" / "MITAboveFiveK"
    default_xmp = repo_root / "outputs" / "xmps"
    default_prev = repo_root / "outputs" / "1_preview"
    default_params = repo_root / "outputs" / "1_data"
    default_luts = repo_root / "outputs" / "luts"
    default_manifest = repo_root / "outputs" / "dataset_manifest.jsonl"

    ap = argparse.ArgumentParser(description="Fit PV2012 sliders to FiveK expert TIFs and write XMP")
    ap.add_argument("--root", default=str(default_root), help="Dataset root (default: data/MITAboveFiveK)")
    ap.add_argument("--split", default="training", choices=["training","validation","testing","debugging"],
                    help="Which split to process if JSON is available (default: training)")
    ap.add_argument("--expert", default="c", help="Expert a|b|c|d|e (default: c)")
    ap.add_argument("--out_xmp_dir", default=str(default_xmp), help="Output XMP dir (default: outputs/xmps)")
    ap.add_argument("--out_preview_dir", default=str(default_prev), help="Output preview dir (default: outputs/1_preview)")
    ap.add_argument("--device", default="mps", help="mps|cuda|cpu (default: mps)")
    ap.add_argument("--long_side", type=int, default=512, help="thumbnail long side (px)")
    ap.add_argument("--steps", type=int, default=800, help="optimization steps per image")
    ap.add_argument("--lr", type=float, default=5e-2, help="optimizer LR")
    ap.add_argument("--limit", type=int, default=0, help="limit number of pairs (0=all)")
    ap.add_argument("--debug_previews", action="store_true", help="save edited preview and target TIFF debug PNGs")
    ap.add_argument("--log_every", type=int, default=0, help="print loss every N steps (0=off)")
    ap.add_argument("--curve_knots", type=int, default=16,
                    help="number of tone-curve knots (higher = more flexibility; default 16)")
    ap.add_argument("--export_json", dest="export_json", action="store_true", default=True,
                    help="also export fitted params as JSON (default: enabled)")
    ap.add_argument("--no-export_json", dest="export_json", action="store_false",
                    help="disable JSON export")
    ap.add_argument("--export_cube", action="store_true", help="also export baked 3D LUT in linear ProPhoto")
    ap.add_argument("--out_params_dir", default=str(default_params), help="Output JSON dir (default: outputs/1_data)")
    ap.add_argument("--out_lut_dir", default=str(default_luts), help="Output 3D LUT dir")
    ap.add_argument("--lut_size", type=int, default=33, help="3D LUT size (e.g., 33)")
    ap.add_argument("--out_dataset_manifest", default=str(default_manifest),
                    help="Path to JSONL manifest mapping ids to 1_preview/1_data (default: outputs/dataset_manifest.jsonl)")
    args = ap.parse_args()

    # device selection (MPS, CUDA, or CPU)
    if args.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    root = Path(args.root)
    out_xmp_dir = Path(args.out_xmp_dir); out_xmp_dir.mkdir(parents=True, exist_ok=True)
    out_prev_dir = Path(args.out_preview_dir); out_prev_dir.mkdir(parents=True, exist_ok=True)
    out_params_dir = Path(args.out_params_dir); out_params_dir.mkdir(parents=True, exist_ok=True)
    out_lut_dir = Path(args.out_lut_dir); out_lut_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: List[Dict[str, object]] = []

    pairs = load_split_paths(root, args.expert, args.split)
    if args.limit > 0:
        pairs = pairs[:args.limit]
    if not pairs:
        print("[ERR] no RAW/TIF pairs found. Check --root and expert/split.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] device={device}, pairs={len(pairs)}")
    for raw_path, tif_path in tqdm(pairs, desc="Fitting", unit="img"):
        try:
            params_denorm, params_norm, edited_preview = fit_params_for_pair(
                raw_path, tif_path, device,
                long_side=args.long_side,
                steps=args.steps, lr=args.lr,
                log_every=args.log_every,
                curve_knots=args.curve_knots,
            )

            # Write XMP next to RAW name (in designated output dir)
            # Be explicit about PV2012 so ACR uses the right process engine.
            xmp_str = xmp_from_params(params_denorm, camera_profile="Adobe Color",
                                      process_version="11.0", version=None)
            xmp_out = out_xmp_dir / f"{raw_path.stem}.xmp"
            xmp_out.write_text(xmp_str, encoding="utf-8")

            # Baseline (unedited) preview PNG
            raw_thumb = read_raw_linear_prophoto(raw_path, long_side=args.long_side)
            raw_preview01 = prophoto_to_srgb_preview(raw_thumb)
            raw_img = (np.clip(raw_preview01 * 255.0, 0, 255)).astype(np.uint8)
            preview_out = out_prev_dir / f"{raw_path.stem}.png"
            iio.imwrite(preview_out, raw_img)

            json_out_path: Optional[Path] = None
            if args.export_json:
                json_out_path = out_params_dir / f"{raw_path.stem}.json"
                export_params_json(params_denorm, params_norm, json_out_path)
            if args.export_cube:
                cube_out = out_lut_dir / f"{raw_path.stem}_prophoto_lin_size{args.lut_size}.cube"
                export_lut3d_cube_prophoto_linear(params_denorm, size=args.lut_size, out_path=cube_out)

            if args.debug_previews:
                # Save edited preview and target TIFF thumbnails for inspection
                edit_img = (np.clip(edited_preview * 255.0, 0, 255)).astype(np.uint8)
                iio.imwrite(out_prev_dir / f"{raw_path.stem}__edit_srgb.png", edit_img)

                y_png01 = read_tif16_color_managed_to_srgb_png(tif_path, long_side=args.long_side)
                y_png = (np.clip(y_png01 * 255.0, 0, 255)).astype(np.uint8)
                iio.imwrite(out_prev_dir / f"{raw_path.stem}__tif_srgb.png", y_png)

            if json_out_path is not None:
                manifest_records.append({
                    "id": raw_path.stem,
                    "split": args.split,
                    "expert": args.expert,
                    "raw": raw_path,
                    "tif": tif_path,
                    "1_preview": preview_out,
                    "1_data": json_out_path,
                })
        except Exception as e:
            print(f"[warn] failed on {raw_path.name}: {e}", file=sys.stderr)

    if manifest_records:
        manifest_path = Path(args.out_dataset_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        def _manifest_path(path: Path) -> str:
            try:
                return str(path.resolve().relative_to(manifest_path.parent.resolve()))
            except ValueError:
                return str(path.resolve())

        with manifest_path.open("w", encoding="utf-8") as mf:
            for record in manifest_records:
                serializable = {
                    "id": record["id"],
                    "split": record["split"],
                    "expert": record["expert"],
                    "raw": _manifest_path(record["raw"]),
                    "tif": _manifest_path(record["tif"]),
                    "1_preview": _manifest_path(record["1_preview"]),
                    "1_data": _manifest_path(record["1_data"]),
                }
                mf.write(json.dumps(serializable) + "\n")

if __name__ == "__main__":
    main()
