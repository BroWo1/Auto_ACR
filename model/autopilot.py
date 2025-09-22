"""Single-encoder multi-head regressor for Auto ACR."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn

from model.backbones import BackboneConfig, ImageBackbone
from model.heads import SlidersHead, ToneCurveHead
from params.ranges import SCALAR_NAMES, TONECURVE_LENGTH


class MetadataEncoder(nn.Module):
    """Embed EXIF-style metadata into a compact feature vector."""

    def __init__(
        self,
        output_dim: int = 128,
        camera_vocab_size: int = 512,
        camera_embed_dim: int = 64,
        mlp_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.camera_embed = nn.Embedding(camera_vocab_size, camera_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(camera_embed_dim + mlp_hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        metadata: Optional[Mapping[str, torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if metadata is None:
            return None
        device = self.camera_embed.weight.device
        batch_size = batch_size or self._infer_batch_size(metadata)
        if batch_size is None:
            return None

        def _to_tensor(name: str, dtype: torch.dtype, default: float = 0.0) -> torch.Tensor:
            value = metadata.get(name)
            if value is None:
                tensor = torch.full((batch_size,), float(default), device=device, dtype=dtype)
            else:
                tensor = torch.as_tensor(value, device=device, dtype=dtype)
                if tensor.ndim == 0:
                    tensor = tensor.expand(batch_size)
                tensor = tensor.to(device=device, dtype=dtype)
            return tensor

        iso = torch.clamp(_to_tensor("iso", torch.float32, 100.0), min=1.0)
        shutter = torch.clamp(_to_tensor("shutter", torch.float32, 1 / 60.0), min=1e-4)
        aperture = torch.clamp(_to_tensor("aperture", torch.float32, 4.0), min=1e-3)

        iso_feat = torch.log(iso)
        shutter_feat = torch.log(shutter)
        aperture_feat = aperture
        exif_feat = torch.stack([iso_feat, shutter_feat, aperture_feat], dim=-1)
        exif_feat = self.mlp(exif_feat)

        camera_id = _to_tensor("camera_id", torch.long, 0)
        camera_feat = self.camera_embed(camera_id)

        fused = torch.cat([camera_feat, exif_feat], dim=-1)
        return self.proj(fused)

    @staticmethod
    def _infer_batch_size(metadata: Mapping[str, torch.Tensor]) -> Optional[int]:
        for value in metadata.values():
            if value is None:
                continue
            tensor = torch.as_tensor(value)
            if tensor.ndim == 0:
                continue
            return tensor.shape[0]
        return None


class FiLMFusion(nn.Module):
    def __init__(self, feature_dim: int, metadata_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or metadata_dim
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )

    def forward(self, features: torch.Tensor, metadata_features: torch.Tensor) -> torch.Tensor:
        scale_shift = self.net(metadata_features)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return features * (scale + 1.0) + shift


class ConcatFusion(nn.Module):
    def __init__(self, feature_dim: int, metadata_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim + metadata_dim, feature_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor, metadata_features: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([features, metadata_features], dim=-1)
        return self.proj(fused)


class AutoPilotRegressor(nn.Module):
    """Single-encoder multi-head regressor returning normalized parameters."""

    def __init__(
        self,
        backbone: BackboneConfig | None = None,
        metadata_encoder: Optional[MetadataEncoder] = None,
        fusion: str = "film",
        tone_curve_hidden: Iterable[int] | None = None,
        sliders_hidden: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = ImageBackbone(backbone)
        self.metadata_encoder = metadata_encoder

        feature_dim = self.backbone.feature_dim
        metadata_dim = metadata_encoder.output_dim if metadata_encoder is not None else 0
        if metadata_encoder and fusion == "film":
            self.fusion = FiLMFusion(feature_dim, metadata_dim)
        elif metadata_encoder and fusion == "concat":
            self.fusion = ConcatFusion(feature_dim, metadata_dim)
        else:
            self.fusion = None
        self.fusion_mode = fusion

        sliders_hidden = tuple(sliders_hidden or (256, 128))
        tone_curve_hidden = tuple(tone_curve_hidden or (256, 128))
        self.sliders_head = SlidersHead(feature_dim, sliders_hidden)
        self.tone_head = ToneCurveHead(feature_dim, tone_curve_hidden)

    def forward(
        self,
        images: torch.Tensor,
        metadata: Optional[Mapping[str, torch.Tensor]] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        _, pooled = self.backbone(images)
        conditioning = None
        if self.metadata_encoder is not None:
            conditioning = self.metadata_encoder(metadata, batch_size=pooled.shape[0])
            if conditioning is not None and conditioning.shape[0] != pooled.shape[0]:
                raise ValueError("Metadata batch dimension does not match image batch")
        fused = pooled
        if conditioning is not None and self.fusion is not None:
            fused = self.fusion(pooled, conditioning)
        elif conditioning is not None:
            fused = pooled + conditioning

        sliders = self.sliders_head(fused)
        tone_curve = self.tone_head(fused)
        if not return_dict:
            return torch.cat([sliders, tone_curve], dim=-1)

        params: Dict[str, torch.Tensor] = {}
        for idx, name in enumerate(SCALAR_NAMES):
            params[name] = sliders[:, idx]
        params["ToneCurve"] = tone_curve
        return {"params_norm": params}

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad_(True)


__all__ = ["AutoPilotRegressor", "MetadataEncoder", "BackboneConfig"]
