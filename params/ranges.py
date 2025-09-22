"""Utilities for normalizing and denormalizing ACR-style parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping


@dataclass(frozen=True)
class SliderRange:
    name: str
    minimum: float
    maximum: float
    default: float = 0.0

    def normalize(self, value: float) -> float:
        span = self.maximum - self.minimum
        if span <= 0:
            raise ValueError(f"Invalid span for {self.name}: {self.minimum}..{self.maximum}")
        norm = 2.0 * ((value - self.minimum) / span) - 1.0
        return _clamp(norm, -1.0, 1.0)

    def denormalize(self, value: float) -> float:
        value = _clamp(value, -1.0, 1.0)
        span = self.maximum - self.minimum
        return self.minimum + (value + 1.0) * 0.5 * span


SCALAR_RANGES: Dict[str, SliderRange] = {
    "Temperature": SliderRange("Temperature", minimum=2000.0, maximum=50000.0, default=6500.0),
    "Tint": SliderRange("Tint", minimum=-150.0, maximum=150.0, default=0.0),
    "Exposure2012": SliderRange("Exposure2012", minimum=-5.0, maximum=5.0, default=0.0),
    "Contrast2012": SliderRange("Contrast2012", minimum=-100.0, maximum=100.0, default=0.0),
    "Highlights2012": SliderRange("Highlights2012", minimum=-100.0, maximum=100.0, default=0.0),
    "Shadows2012": SliderRange("Shadows2012", minimum=-100.0, maximum=100.0, default=0.0),
    "Whites2012": SliderRange("Whites2012", minimum=-100.0, maximum=100.0, default=0.0),
    "Blacks2012": SliderRange("Blacks2012", minimum=-100.0, maximum=100.0, default=0.0),
    "Vibrance": SliderRange("Vibrance", minimum=-100.0, maximum=100.0, default=0.0),
    "Saturation": SliderRange("Saturation", minimum=-100.0, maximum=100.0, default=0.0),
}


SCALAR_NAMES: tuple[str, ...] = tuple(SCALAR_RANGES.keys())
TONECURVE_LENGTH: int = 16


def normalize_scalars(params: Mapping[str, float]) -> Dict[str, float]:
    """Normalize the scalar parameters into [-1, 1]."""
    normalized: Dict[str, float] = {}
    for name in SCALAR_NAMES:
        if name not in params:
            value = SCALAR_RANGES[name].default
        else:
            value = float(params[name])
        normalized[name] = SCALAR_RANGES[name].normalize(value)
    return normalized


def denormalize_scalars(params: Mapping[str, float]) -> Dict[str, float]:
    """Denormalize the scalar parameters into human-friendly units."""
    denormalized: Dict[str, float] = {}
    for name in SCALAR_NAMES:
        if name not in params:
            value = 0.0
        else:
            value = float(params[name])
        denormalized[name] = SCALAR_RANGES[name].denormalize(value)
    return denormalized


def normalize_tone_curve(curve: Iterable[float]) -> list[float]:
    """Return a monotone tone curve clipped to [0, 1] with the expected length."""
    values = list(curve)
    if len(values) != TONECURVE_LENGTH:
        raise ValueError(f"Tone curve must contain {TONECURVE_LENGTH} values; got {len(values)}")
    monotone = []
    last = 0.0
    for value in values:
        v = max(float(value), last)
        monotone.append(v)
        last = v
    if monotone[-1] <= 0.0:
        raise ValueError("Tone curve is degenerate; last value must be > 0")
    scale = monotone[-1]
    normalized = [v / scale for v in monotone]
    normalized[0] = 0.0
    normalized[-1] = 1.0
    return normalized


def denormalize_tone_curve(curve: Iterable[float]) -> list[float]:
    """Ensure a normalized tone curve lives in [0, 1] and is strictly increasing."""
    values = list(curve)
    if len(values) != TONECURVE_LENGTH:
        raise ValueError(f"Tone curve must contain {TONECURVE_LENGTH} values; got {len(values)}")
    snapped = []
    last = 0.0
    for idx, value in enumerate(values):
        v = _clamp(float(value), 0.0, 1.0)
        if idx == 0:
            v = 0.0
        elif idx == len(values) - 1:
            v = 1.0
        if v < last:
            v = last
        snapped.append(v)
        last = v
    if snapped[-1] < 1.0:
        snapped[-1] = 1.0
    return snapped


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
