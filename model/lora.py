"""Lightweight LoRA helpers for personalising the AutoPilot regressor."""
from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    target_patterns: Sequence[str] = ("blocks.*.attn.qkv", "blocks.*.attn.proj", "blocks.*.mlp.fc1", "blocks.*.mlp.fc2")


class LoRALinear(nn.Module):
    """Wrap an existing linear layer with a low-rank residual."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)
        self.lora_down = nn.Parameter(torch.zeros(linear.in_features, rank))
        self.lora_up = nn.Parameter(torch.zeros(rank, linear.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        update = x @ self.lora_down @ self.lora_up
        return base + self.scaling * update

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


def inject_lora(model: nn.Module, config: LoRAConfig | None = None, modules: Iterable[str] | None = None) -> List[str]:
    """Replace target linear layers with LoRA-wrapped equivalents.

    Args:
        model: Module whose submodules should receive LoRA adapters.
        config: Configuration specifying rank/alpha and patterns.
        modules: Explicit list of module names to adapt. If provided it overrides patterns.

    Returns:
        Names of the modules that were replaced.
    """
    config = config or LoRAConfig()
    target_names: List[str]
    if modules is not None:
        target_names = list(modules)
    else:
        target_names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and any(fnmatch.fnmatch(name, pattern) for pattern in config.target_patterns)
        ]
    replaced = []
    module_lookup = dict(model.named_modules())
    for name in target_names:
        linear = module_lookup.get(name)
        if not isinstance(linear, nn.Linear):
            continue
        parent, attr = _split_parent(model, name)
        setattr(parent, attr, LoRALinear(linear, rank=config.rank, alpha=config.alpha))
        replaced.append(name)
    return replaced


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield from module.parameters()


def _split_parent(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


__all__ = ["LoRAConfig", "LoRALinear", "inject_lora", "lora_parameters"]
