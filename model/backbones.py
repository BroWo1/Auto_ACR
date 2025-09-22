"""Backbone encoder utilities for the Auto ACR model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn


@dataclass
class BackboneConfig:
    name: str = "vit_small_patch16_224"
    pretrained: bool = False
    drop_path_rate: float = 0.1
    global_pool: str = "token"


class ImageBackbone(nn.Module):
    """Thin wrapper around `timm` encoders that exposes pooled features."""

    def __init__(self, config: BackboneConfig | None = None):
        super().__init__()
        self.config = config or BackboneConfig()
        self.encoder = timm.create_model(
            self.config.name,
            pretrained=self.config.pretrained,
            num_classes=0,
            drop_path_rate=self.config.drop_path_rate,
        )
        self.feature_dim = getattr(self.encoder, "num_features", None)
        if self.feature_dim is None:
            raise ValueError(
                f"Backbone {self.config.name} does not expose `num_features`. "
                "Please choose a timm model with a known feature dimension."
            )
        self.global_pool = self.config.global_pool

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return token-level features and a pooled descriptor."""
        feats = self.encoder.forward_features(images)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        pooled = self._pool(feats)
        return feats, pooled

    def _pool(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.ndim == 3:
            if self.global_pool == "token":
                return feats[:, 0]
            if self.global_pool == "mean":
                return feats.mean(dim=1)
            raise ValueError(f"Unsupported global_pool for token features: {self.global_pool}")
        if feats.ndim == 4:
            if self.global_pool == "mean":
                return feats.mean(dim=(2, 3))
            if self.global_pool == "token":
                raise ValueError("token pooling not supported for convolutional backbones")
            raise ValueError(f"Unsupported global_pool for map features: {self.global_pool}")
        raise ValueError(f"Unexpected feature tensor shape: {feats.shape}")


def create_backbone(name: str, pretrained: bool = True, drop_path_rate: float = 0.1, global_pool: str = "token") -> ImageBackbone:
    config = BackboneConfig(name=name, pretrained=pretrained, drop_path_rate=drop_path_rate, global_pool=global_pool)
    return ImageBackbone(config)
