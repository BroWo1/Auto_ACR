"""Differentiable approximation of an ACR-style edit pipeline."""
from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from params.ranges import SCALAR_RANGES, SCALAR_NAMES, TONECURVE_LENGTH


def _denormalize_slider(name: str, value: torch.Tensor) -> torch.Tensor:
    rng = SCALAR_RANGES[name]
    span = rng.maximum - rng.minimum
    return rng.minimum + 0.5 * (value + 1.0) * span


def _luminance(image: torch.Tensor) -> torch.Tensor:
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _apply_white_balance(image: torch.Tensor, temperature: torch.Tensor, tint: torch.Tensor) -> torch.Tensor:
    reference = 6500.0
    temp = torch.clamp(temperature, min=2000.0, max=50000.0)
    temp_ratio = reference / temp
    red_gain = temp_ratio.sqrt()
    blue_gain = temp_ratio.rsqrt()
    tint_scale = torch.clamp(tint / 150.0, min=-1.5, max=1.5)
    green_gain = 1.0 - 0.1 * tint_scale
    red_gain = red_gain * (1.0 + 0.05 * tint_scale)
    blue_gain = blue_gain * (1.0 - 0.05 * tint_scale)
    gains = torch.stack([red_gain, green_gain, blue_gain], dim=1).unsqueeze(-1).unsqueeze(-1)
    balanced = image * gains
    norm = gains.max(dim=1, keepdim=True)[0]
    balanced = balanced / torch.clamp(norm, min=1e-4)
    return balanced.clamp(0.0, 4.0)


def _apply_exposure(image: torch.Tensor, exposure: torch.Tensor) -> torch.Tensor:
    gain = torch.pow(2.0, exposure).view(-1, 1, 1, 1)
    return image * gain


def _apply_contrast(image: torch.Tensor, contrast: torch.Tensor) -> torch.Tensor:
    factor = (1.0 + contrast / 100.0).view(-1, 1, 1, 1)
    midpoint = 0.5
    return ((image - midpoint) * factor + midpoint).clamp(0.0, 1.0)


def _apply_region_adjustment(
    image: torch.Tensor,
    luma: torch.Tensor,
    value: torch.Tensor,
    pivot: float,
    width: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    strength = value / 100.0
    if torch.allclose(strength, torch.zeros_like(strength)):
        return image, luma
    mask = torch.sigmoid((luma - pivot) / width)
    mask = mask - mask.mean(dim=(2, 3), keepdim=True)
    delta = strength.view(-1, 1, 1, 1) * mask
    new_luma = torch.clamp(luma + delta, 0.0, 1.0)
    ratio = torch.where(
        luma > 1e-4,
        new_luma / torch.clamp(luma, min=1e-4),
        torch.ones_like(luma),
    )
    adjusted = torch.clamp(image * ratio, 0.0, 1.0)
    return adjusted, new_luma


def _apply_tone_curve(
    image: torch.Tensor,
    tone_curve: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    if tone_curve.shape[-1] != TONECURVE_LENGTH:
        raise ValueError(f"Tone curve must have length {TONECURVE_LENGTH}")
    luma = torch.clamp(_luminance(image), 0.0, 1.0)
    curve = F.interpolate(
        tone_curve.unsqueeze(1),
        size=resolution,
        mode="linear",
        align_corners=True,
    ).squeeze(1)
    coords = luma * (resolution - 1)
    idx_low = torch.floor(coords).long()
    idx_high = torch.clamp(idx_low + 1, max=resolution - 1)
    weight = (coords - idx_low.float()).view(coords.shape)
    batch = image.shape[0]
    flat_low = idx_low.view(batch, -1)
    flat_high = idx_high.view(batch, -1)
    gathered_low = torch.gather(curve, 1, flat_low).view_as(luma)
    gathered_high = torch.gather(curve, 1, flat_high).view_as(luma)
    target_luma = (1.0 - weight) * gathered_low + weight * gathered_high
    ratio = torch.where(
        luma > 1e-5,
        target_luma / torch.clamp(luma, min=1e-5),
        torch.ones_like(luma),
    )
    return torch.clamp(image * ratio, 0.0, 1.0)


def _apply_saturation(image: torch.Tensor, saturation: torch.Tensor) -> torch.Tensor:
    luma = _luminance(image)
    chroma = image - luma
    gain = (1.0 + saturation / 100.0).view(-1, 1, 1, 1)
    adjusted = torch.clamp(luma + chroma * gain, 0.0, 1.0)
    return adjusted


def _apply_vibrance(image: torch.Tensor, vibrance: torch.Tensor) -> torch.Tensor:
    luma = _luminance(image)
    chroma = image - luma
    chroma_norm = torch.sqrt(torch.sum(chroma * chroma, dim=1, keepdim=True) + 1e-6)
    mask = torch.exp(-4.0 * chroma_norm)
    gain = (1.0 + (vibrance / 100.0).view(-1, 1, 1, 1) * mask).clamp(0.2, 4.0)
    adjusted = torch.clamp(luma + chroma * gain, 0.0, 1.0)
    return adjusted


class DifferentiableEditLayer(nn.Module):
    def __init__(self, tone_curve_resolution: int = 1024) -> None:
        super().__init__()
        self.tone_curve_resolution = tone_curve_resolution

    def forward(
        self,
        image: torch.Tensor,
        params_norm: Mapping[str, torch.Tensor],
        tone_curve: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image = torch.clamp(image, 0.0, 1.0)
        batch = image.shape[0]
        scalars = {}
        for name in SCALAR_NAMES:
            value = params_norm.get(name)
            if value is None:
                scalars[name] = torch.zeros(batch, device=image.device, dtype=image.dtype)
            elif value.ndim == 0:
                scalars[name] = value.view(1).expand(batch)
            else:
                scalars[name] = value
        if tone_curve is None:
            tone_curve = params_norm.get("ToneCurve")
        if tone_curve is None:
            raise ValueError("ToneCurve missing from parameters")
        if tone_curve.ndim == 1:
            tone_curve = tone_curve.view(1, -1).expand(batch, -1)

        temperature = _denormalize_slider("Temperature", scalars["Temperature"])
        tint = _denormalize_slider("Tint", scalars["Tint"])
        exposure = _denormalize_slider("Exposure2012", scalars["Exposure2012"])
        contrast = _denormalize_slider("Contrast2012", scalars["Contrast2012"])
        highlights = _denormalize_slider("Highlights2012", scalars["Highlights2012"])
        shadows = _denormalize_slider("Shadows2012", scalars["Shadows2012"])
        whites = _denormalize_slider("Whites2012", scalars["Whites2012"])
        blacks = _denormalize_slider("Blacks2012", scalars["Blacks2012"])
        vibrance = _denormalize_slider("Vibrance", scalars["Vibrance"])
        saturation = _denormalize_slider("Saturation", scalars["Saturation"])

        edited = _apply_white_balance(image, temperature, tint)
        edited = _apply_exposure(edited, exposure)
        edited = edited.clamp(0.0, 4.0)
        edited = _apply_contrast(edited, contrast)

        luma = torch.clamp(_luminance(edited), 0.0, 1.0)
        edited, luma = _apply_region_adjustment(edited, luma, highlights, pivot=0.7, width=0.1)
        edited, luma = _apply_region_adjustment(edited, luma, shadows, pivot=0.3, width=0.12)
        edited, luma = _apply_region_adjustment(edited, luma, whites, pivot=0.9, width=0.08)
        edited, luma = _apply_region_adjustment(edited, luma, blacks, pivot=0.1, width=0.08)

        edited = _apply_tone_curve(edited, tone_curve, self.tone_curve_resolution)
        edited = _apply_vibrance(edited, vibrance)
        edited = _apply_saturation(edited, saturation)
        return edited.clamp(0.0, 1.0)


__all__ = ["DifferentiableEditLayer"]
