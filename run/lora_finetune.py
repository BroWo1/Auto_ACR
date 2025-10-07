#!/usr/bin/env python3
"""LoRA fine-tuning for the AutoPilot regressor using slider/tone JSON supervision."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.autopilot import AutoPilotRegressor, BackboneConfig
from model.lora import LoRAConfig, LoRALinear, inject_lora, lora_parameters
from params.ranges import SCALAR_NAMES
from run.train_autopilot import (  # type: ignore
    PreviewParamsDataset,
    collate_fn,
    filter_entries,
    load_manifest,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Dataset manifest JSONL")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Base model checkpoint (.pt)")
    parser.add_argument("--output", type=Path, required=True, help="Directory for LoRA checkpoint")
    parser.add_argument("--train-split", default="training", help="Manifest split for training")
    parser.add_argument("--val-split", default="", help="Optional validation split name")
    parser.add_argument("--expert", default=None, help="Optional manifest expert filter")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tone-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="Device spec: auto|cuda|cpu|mps")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--imagenet-norm", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-target", nargs="*", default=None,
                        help="Module name patterns to adapt (overrides defaults)")
    parser.add_argument("--train-heads", action="store_true", help="Also fine-tune slider/tone heads")
    parser.add_argument("--backbone", default="vit_small_patch16_224")
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--save-frequency", type=int, default=0, help="Optional epoch frequency for intermediate saves")
    return parser.parse_args()


def select_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def build_model(backbone: str, drop_path: float, pretrained: bool) -> AutoPilotRegressor:
    backbone_cfg = BackboneConfig(name=backbone, drop_path_rate=drop_path, pretrained=pretrained)
    return AutoPilotRegressor(backbone=backbone_cfg, metadata_encoder=None, fusion="film")


def load_base_model(checkpoint_path: Path, backbone: str, drop_path: float, pretrained_backbone: bool,
                    device: torch.device) -> tuple[AutoPilotRegressor, Dict]:
    model = build_model(backbone, drop_path, pretrained_backbone)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, ckpt


def freeze_base_params(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def collect_trainable_params(model: nn.Module, train_heads: bool) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for param in lora_parameters(model):
        param.requires_grad_(True)
        params.append(param)
    if train_heads:
        for param in model.sliders_head.parameters():
            param.requires_grad_(True)
            params.append(param)
        for param in model.tone_head.parameters():
            param.requires_grad_(True)
            params.append(param)
    return params


def extract_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_down"] = module.lora_down.detach().cpu()
            state[f"{name}.lora_up"] = module.lora_up.detach().cpu()
            state[f"{name}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
            state[f"{name}.rank"] = torch.tensor(module.rank, dtype=torch.int32)
    return state


def run_epoch(model: AutoPilotRegressor, loader: DataLoader, device: torch.device,
              optimizer: Optional[torch.optim.Optimizer], tone_weight: float, train: bool) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_scalar = 0.0
    total_tone = 0.0
    batches = 0

    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            scalars_gt = batch["scalars"].to(device, non_blocking=True)
            tone_gt = batch["tone"].to(device, non_blocking=True)

            outputs = model(images)["params_norm"]
            scalars_pred = torch.stack([outputs[name] for name in SCALAR_NAMES], dim=1)
            tone_pred = outputs["ToneCurve"]

            scalar_loss = F.mse_loss(scalars_pred, scalars_gt)
            tone_loss = F.mse_loss(tone_pred, tone_gt)
            loss = scalar_loss + tone_weight * tone_loss

            if train:
                optimizer.zero_grad(set_to_none=True)  # type: ignore
                loss.backward()
                optimizer.step()  # type: ignore

            total_loss += loss.item()
            total_scalar += scalar_loss.item()
            total_tone += tone_loss.item()
            batches += 1

    denom = max(batches, 1)
    return {
        "loss": total_loss / denom,
        "scalar_loss": total_scalar / denom,
        "tone_loss": total_tone / denom,
    }


def save_checkpoint(model: AutoPilotRegressor, lora_config: LoRAConfig, base_checkpoint: Path,
                    output_dir: Path, epoch: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_state = extract_lora_state(model)
    payload = {
        "lora_state": lora_state,
        "lora_config": {
            "rank": lora_config.rank,
            "alpha": lora_config.alpha,
            "target_patterns": list(lora_config.target_patterns),
        },
        "base_checkpoint": str(base_checkpoint),
        "epoch": epoch,
    }
    torch.save(payload, output_dir / f"lora_epoch{epoch:03d}.pt")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    entries = load_manifest(args.manifest)
    train_entries = filter_entries(entries, args.train_split, args.expert, limit=0)
    val_entries: Optional[List[Dict[str, str]]] = None
    if args.val_split:
        try:
            val_entries = filter_entries(entries, args.val_split, args.expert, limit=0)
        except ValueError:
            val_entries = None

    train_dataset = PreviewParamsDataset(
        train_entries,
        image_size=args.image_size,
        augment=not args.no_augment,
        imagenet_norm=args.imagenet_norm,
        load_raw_images=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=device.type == "cuda", collate_fn=collate_fn)

    val_loader = None
    if val_entries:
        val_dataset = PreviewParamsDataset(
            val_entries,
            image_size=args.image_size,
            augment=False,
            imagenet_norm=args.imagenet_norm,
            load_raw_images=False,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=device.type == "cuda", collate_fn=collate_fn)

    model, base_ckpt = load_base_model(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        drop_path=args.drop_path,
        pretrained_backbone=args.pretrained_backbone,
        device=device,
    )

    lora_config = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha,
                             target_patterns=tuple(args.lora_target) if args.lora_target else LoRAConfig().target_patterns)
    replaced = inject_lora(model.backbone, lora_config, modules=None)
    print(f"[info] Injected LoRA into {len(replaced)} modules")

    # Ensure newly created LoRA parameters live on the training device
    model.to(device)

    freeze_base_params(model)
    trainable_params = list(collect_trainable_params(model, args.train_heads))
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected for LoRA fine-tuning")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, device, optimizer, args.tone_weight, train=True)
        log = f"[epoch {epoch}] train_loss={train_stats['loss']:.5f}"
        if val_loader is not None:
            val_stats = run_epoch(model, val_loader, device, optimizer=None, tone_weight=args.tone_weight, train=False)
            log += f" val_loss={val_stats['loss']:.5f}"
        print(log)

        if args.save_frequency and (epoch % args.save_frequency == 0 or epoch == args.epochs):
            save_checkpoint(model, lora_config, args.checkpoint, args.output, epoch)

    if not args.save_frequency:
        save_checkpoint(model, lora_config, args.checkpoint, args.output, args.epochs)


if __name__ == "__main__":
    main()
