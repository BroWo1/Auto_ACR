"""Prediction heads for sliders and tone curve."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from params.ranges import SCALAR_NAMES, TONECURVE_LENGTH


class SlidersHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 128)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.GELU())
            in_dim = dim
        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(in_dim, len(SCALAR_NAMES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.mlp(features)
        return torch.tanh(self.proj(x))


class ToneCurveHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 128), epsilon: float = 1e-4):
        super().__init__()
        layers = []
        in_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.GELU())
            in_dim = dim
        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(in_dim, TONECURVE_LENGTH)
        self.epsilon = epsilon

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.proj(self.mlp(features))
        deltas = F.softplus(raw) + self.epsilon
        curve = torch.cumsum(deltas, dim=-1)
        start = curve[:, :1]
        end = curve[:, -1:]
        normalized = (curve - start) / (end - start + 1e-6)
        normalized = normalized.clamp(0.0, 1.0)
        normalized = normalized.clone()
        normalized[:, 0] = 0.0
        normalized[:, -1] = 1.0
        return normalized
