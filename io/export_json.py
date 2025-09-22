"""Exporter for AutoPilot model predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from params.ranges import (
    SCALAR_NAMES,
    TONECURVE_LENGTH,
    denormalize_scalars,
    denormalize_tone_curve,
)


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        raise ValueError("Tensor values must be scalar before export")
    return float(value)


def _to_list(values: Any) -> list[float]:
    if isinstance(values, torch.Tensor):
        if values.ndim == 1:
            return [float(v) for v in values.detach().cpu().tolist()]
        raise ValueError("Tone curve tensor must be 1-D")
    return [float(v) for v in values]


def build_export_dict(
    params_norm: Mapping[str, Any],
    format_name: str = "acr-autopilot-v1",
) -> dict[str, Any]:
    scalars_norm = {name: _to_float(params_norm[name]) for name in SCALAR_NAMES}
    tone_curve_norm = _to_list(params_norm["ToneCurve"])
    if len(tone_curve_norm) != TONECURVE_LENGTH:
        raise ValueError(f"ToneCurve must have {TONECURVE_LENGTH} elements; got {len(tone_curve_norm)}")

    scalars_denorm = denormalize_scalars(scalars_norm)
    tone_curve_denorm = denormalize_tone_curve(tone_curve_norm)

    export = {
        "format": format_name,
        "params_norm": {**scalars_norm, "ToneCurve": tone_curve_norm},
        "params_denorm": {**scalars_denorm, "ToneCurve": tone_curve_denorm},
    }
    return export


def export_to_json(
    params_norm: Mapping[str, Any],
    path: Path | str,
    format_name: str = "acr-autopilot-v1",
    indent: int = 2,
) -> None:
    export_dict = build_export_dict(params_norm, format_name=format_name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(export_dict, fp, indent=indent)


__all__ = ["export_to_json", "build_export_dict"]
