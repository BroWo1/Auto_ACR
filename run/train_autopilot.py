#!/usr/bin/env python3
"""Train the AutoPilot regressor on preview + JSON parameter pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from model.autopilot import AutoPilotRegressor, BackboneConfig
from model.edit_layer import DifferentiableEditLayer
from params.ranges import SCALAR_NAMES, TONECURVE_LENGTH, normalize_scalars, normalize_tone_curve

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/dataset_manifest.jsonl"),
                        help="Path to JSONL manifest produced by fivek_tif_to_xmp.py")
    parser.add_argument("--train-split", default="training", help="Manifest split name to use for training")
    parser.add_argument("--val-split", default="validation", help="Optional split name for validation (blank to disable)")
    parser.add_argument("--expert", default=None, help="Filter manifest entries to a specific expert (e.g. 'c')")
    parser.add_argument("--image-size", type=int, default=224, help="Square input size passed to the backbone")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--tone-weight", type=float, default=1.0, help="Relative loss weight for the tone curve head")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient norm clip value (<=0 to disable)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--device", default="auto", help="Device identifier: auto|cuda|cpu|mps|cuda:0 ...")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--imagenet-norm", action="store_true", help="Apply ImageNet mean/std normalization")
    parser.add_argument("--no-augment", action="store_true", help="Disable random flip augmentation on training data")
    parser.add_argument("--train-limit", type=int, default=0, help="Optional limit on the number of training samples")
    parser.add_argument("--val-limit", type=int, default=0, help="Optional limit on the number of validation samples")
    parser.add_argument("--backbone", default="vit_small_patch16_224", help="timm backbone name")
    parser.add_argument("--pretrained-backbone", action="store_true", help="Use pretrained weights for the backbone")
    parser.add_argument("--drop-path", type=float, default=0.1, help="Drop path rate for the backbone")
    parser.add_argument("--output-dir", type=Path, default=Path("run/checkpoints"), help="Directory to store checkpoints")
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a checkpoint path")
    parser.add_argument("--use-edit-layer", action="store_true", help="Use DifferentiableEditLayer for image-space loss")
    parser.add_argument("--image-loss-weight", type=float, default=0.7, help="Weight for image-space loss (vs param loss)")
    parser.add_argument("--edit-resolution", type=int, default=512, help="Resolution for applying edits (if using edit layer)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def resolve_path(anchor: Path, maybe_path: str) -> Path:
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (anchor / p).resolve()


def load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    entries: List[Dict[str, str]] = []
    parent = manifest_path.parent
    with manifest_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["preview"] = str(resolve_path(parent, data["preview"]))
            data["data"] = str(resolve_path(parent, data["data"]))
            entries.append(data)
    return entries


@dataclass
class Sample:
    image: torch.Tensor
    scalars: torch.Tensor
    tone: torch.Tensor
    raw: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None


class PreviewParamsDataset(Dataset):
    """Dataset combining preview PNGs with normalized parameter targets."""

    def __init__(
        self,
        entries: Iterable[Dict[str, str]],
        image_size: int,
        augment: bool,
        imagenet_norm: bool,
        load_raw_images: bool = False,
        edit_resolution: int = 512,
    ) -> None:
        self.entries: List[Dict[str, str]] = list(entries)
        if not self.entries:
            raise ValueError("Dataset is empty")
        self.transform = self._build_transform(image_size, augment, imagenet_norm)
        self.load_raw_images = load_raw_images
        self.edit_resolution = edit_resolution

    @staticmethod
    def _build_transform(image_size: int, augment: bool, imagenet_norm: bool) -> T.Compose:
        ops: List[T.Transform] = [
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        ]
        if augment:
            ops.append(T.RandomHorizontalFlip())
        ops.append(T.ToTensor())
        if imagenet_norm:
            ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        return T.Compose(ops)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Sample:
        entry = self.entries[idx]
        image = self._load_image(entry["preview"])
        scalars, tone = self._load_targets(entry["data"])

        raw = None
        target = None
        if self.load_raw_images:
            raw = self._load_raw_image(entry.get("raw"))
            target = self._load_target_image(entry.get("tif"))

        return Sample(image=image, scalars=scalars, tone=tone, raw=raw, target=target)

    def _load_image(self, path_str: str) -> torch.Tensor:
        from PIL import Image

        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Preview not found: {path}")
        with Image.open(path) as img:
            image = img.convert("RGB")
            return self.transform(image)

    def _load_targets(self, path_str: str) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Params JSON not found: {path}")
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        params_norm = payload.get("params_norm")
        if params_norm is None:
            params_denorm = payload.get("params_denorm")
            if params_denorm is None:
                raise KeyError("JSON must contain params_norm or params_denorm")
            scalars_norm = normalize_scalars(params_denorm)
            tone_norm = normalize_tone_curve(params_denorm["ToneCurve"])
        else:
            scalars_norm = {name: float(params_norm[name]) for name in SCALAR_NAMES}
            tone_norm = [float(v) for v in params_norm["ToneCurve"]]

        scalars_tensor = torch.tensor([scalars_norm[name] for name in SCALAR_NAMES], dtype=torch.float32)
        if len(tone_norm) != TONECURVE_LENGTH:
            raise ValueError(f"Tone curve length mismatch: expected {TONECURVE_LENGTH}, got {len(tone_norm)}")
        tone_tensor = torch.tensor(tone_norm, dtype=torch.float32)
        return scalars_tensor, tone_tensor

    def _load_raw_image(self, path_str: Optional[str]) -> Optional[torch.Tensor]:
        """Load RAW image in linear ProPhoto RGB for editing."""
        if path_str is None:
            return None
        try:
            import rawpy
            import cv2
            path = Path(path_str)
            if not path.exists():
                return None

            with rawpy.imread(str(path)) as raw:
                rgb16 = raw.postprocess(
                    output_bps=16,
                    no_auto_bright=True,
                    gamma=(1, 1),
                    output_color=rawpy.ColorSpace.ProPhoto,
                    use_camera_wb=True,
                )

            # Resize to edit resolution
            h, w = rgb16.shape[:2]
            scale = self.edit_resolution / max(h, w)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                rgb16 = cv2.resize(rgb16, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert to [C, H, W] float tensor in [0, 1]
            tensor = torch.from_numpy(rgb16).permute(2, 0, 1).float() / 65535.0
            return tensor
        except Exception as e:
            print(f"Warning: Failed to load raw image {path_str}: {e}")
            return None

    def _load_target_image(self, path_str: Optional[str]) -> Optional[torch.Tensor]:
        """Load target TIFF in linear ProPhoto RGB."""
        if path_str is None:
            return None
        try:
            import imageio.v3 as iio
            import cv2
            import numpy as np

            path = Path(path_str)
            if not path.exists():
                return None

            # Read TIFF
            arr = iio.imread(str(path))

            # Resize to edit resolution
            h, w = arr.shape[:2]
            scale = self.edit_resolution / max(h, w)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Normalize to [0, 1]
            if np.issubdtype(arr.dtype, np.floating):
                arr01 = np.clip(arr, 0.0, 1.0).astype(np.float32)
            elif np.issubdtype(arr.dtype, np.integer):
                info = np.iinfo(arr.dtype)
                arr01 = (arr.astype(np.float32) / float(info.max))
            else:
                arr01 = arr.astype(np.float32)
                arr01 = np.clip(arr01 / (arr01.max() if arr01.max() > 0 else 1.0), 0.0, 1.0)

            # Convert to linear ProPhoto (assume gamma 1.8 encoding)
            tensor = torch.from_numpy(arr01).permute(2, 0, 1)
            # Decode gamma 1.8
            thresh = 1.0 / 32.0
            linear = torch.where(
                tensor <= thresh,
                tensor / 16.0,
                tensor.clamp(0, 1) ** 1.8
            )
            return linear
        except Exception as e:
            print(f"Warning: Failed to load target image {path_str}: {e}")
            return None


def collate_fn(batch: List[Sample]) -> Dict[str, torch.Tensor]:
    images = torch.stack([sample.image for sample in batch], dim=0)
    scalars = torch.stack([sample.scalars for sample in batch], dim=0)
    tone = torch.stack([sample.tone for sample in batch], dim=0)

    result = {"image": images, "scalars": scalars, "tone": tone}

    # Add raw and target if available
    if batch[0].raw is not None:
        # Filter out None values and pad to same size
        raw_images = [s.raw for s in batch if s.raw is not None]
        if raw_images:
            # Pad to same size (find max dimensions)
            max_h = max(img.shape[1] for img in raw_images)
            max_w = max(img.shape[2] for img in raw_images)
            padded_raw = []
            for img in raw_images:
                if img.shape[1] < max_h or img.shape[2] < max_w:
                    pad_h = max_h - img.shape[1]
                    pad_w = max_w - img.shape[2]
                    img = F.pad(img, (0, pad_w, 0, pad_h))
                padded_raw.append(img)
            result["raw"] = torch.stack(padded_raw, dim=0)

    if batch[0].target is not None:
        target_images = [s.target for s in batch if s.target is not None]
        if target_images:
            max_h = max(img.shape[1] for img in target_images)
            max_w = max(img.shape[2] for img in target_images)
            padded_target = []
            for img in target_images:
                if img.shape[1] < max_h or img.shape[2] < max_w:
                    pad_h = max_h - img.shape[1]
                    pad_w = max_w - img.shape[2]
                    img = F.pad(img, (0, pad_w, 0, pad_h))
                padded_target.append(img)
            result["target"] = torch.stack(padded_target, dim=0)

    return result


def build_model(args: argparse.Namespace) -> AutoPilotRegressor:
    backbone_cfg = BackboneConfig(
        name=args.backbone,
        pretrained=args.pretrained_backbone,
        drop_path_rate=args.drop_path,
    )
    model = AutoPilotRegressor(backbone=backbone_cfg, metadata_encoder=None, fusion="film")
    return model


def train_one_epoch(
    model: AutoPilotRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    tone_weight: float,
    clip_grad: float,
    epoch: int,
    use_amp: bool,
    edit_layer: Optional[DifferentiableEditLayer] = None,
    image_loss_weight: float = 0.7,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_scalar = 0.0
    total_tone = 0.0
    total_image = 0.0
    num_batches = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        scalars_gt = batch["scalars"].to(device, non_blocking=True)
        tone_gt = batch["tone"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)["params_norm"]
            scalars_pred = torch.stack([outputs[name] for name in SCALAR_NAMES], dim=1)
            tone_pred = outputs["ToneCurve"]

            # Parameter loss
            scalar_loss = F.mse_loss(scalars_pred, scalars_gt)
            tone_loss = F.mse_loss(tone_pred, tone_gt)
            param_loss = scalar_loss + tone_weight * tone_loss

            # Image loss (if edit layer is provided)
            image_loss = torch.tensor(0.0, device=device)
            if edit_layer is not None and "raw" in batch and "target" in batch:
                raw = batch["raw"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)

                # Build params dict for edit layer
                params_dict = {name: outputs[name] for name in SCALAR_NAMES}

                # Apply edits
                edited = edit_layer(raw, params_dict, tone_pred)

                # Compute image space loss
                image_loss = F.mse_loss(edited, target)

            # Combined loss
            if image_loss.item() > 0:
                loss = (1 - image_loss_weight) * param_loss + image_loss_weight * image_loss
            else:
                loss = param_loss

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item()
        total_scalar += scalar_loss.item()
        total_tone += tone_loss.item()
        total_image += image_loss.item()
        num_batches += 1

        postfix = {"loss": loss.item(), "param": param_loss.item()}
        if image_loss.item() > 0:
            postfix["image"] = image_loss.item()
        progress.set_postfix(postfix)

    return {
        "loss": total_loss / max(num_batches, 1),
        "scalar_loss": total_scalar / max(num_batches, 1),
        "tone_loss": total_tone / max(num_batches, 1),
        "image_loss": total_image / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: AutoPilotRegressor,
    loader: DataLoader,
    device: torch.device,
    tone_weight: float,
    epoch: int,
    use_amp: bool,
    edit_layer: Optional[DifferentiableEditLayer] = None,
    image_loss_weight: float = 0.7,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_scalar = 0.0
    total_tone = 0.0
    total_image = 0.0
    num_batches = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        scalars_gt = batch["scalars"].to(device, non_blocking=True)
        tone_gt = batch["tone"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)["params_norm"]
            scalars_pred = torch.stack([outputs[name] for name in SCALAR_NAMES], dim=1)
            tone_pred = outputs["ToneCurve"]

            # Parameter loss
            scalar_loss = F.mse_loss(scalars_pred, scalars_gt)
            tone_loss = F.mse_loss(tone_pred, tone_gt)
            param_loss = scalar_loss + tone_weight * tone_loss

            # Image loss (if edit layer is provided)
            image_loss = torch.tensor(0.0, device=device)
            if edit_layer is not None and "raw" in batch and "target" in batch:
                raw = batch["raw"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)

                # Build params dict for edit layer
                params_dict = {name: outputs[name] for name in SCALAR_NAMES}

                # Apply edits
                edited = edit_layer(raw, params_dict, tone_pred)

                # Compute image space loss
                image_loss = F.mse_loss(edited, target)

            # Combined loss
            if image_loss.item() > 0:
                loss = (1 - image_loss_weight) * param_loss + image_loss_weight * image_loss
            else:
                loss = param_loss

        total_loss += loss.item()
        total_scalar += scalar_loss.item()
        total_tone += tone_loss.item()
        total_image += image_loss.item()
        num_batches += 1

        postfix = {"loss": loss.item(), "param": param_loss.item()}
        if image_loss.item() > 0:
            postfix["image"] = image_loss.item()
        progress.set_postfix(postfix)

    return {
        "loss": total_loss / max(num_batches, 1),
        "scalar_loss": total_scalar / max(num_batches, 1),
        "tone_loss": total_tone / max(num_batches, 1),
        "image_loss": total_image / max(num_batches, 1),
    }


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    stats: Dict[str, float],
    val_stats: Optional[Dict[str, float]],
    best_so_far: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stats": stats,
        "val_stats": val_stats,
    }
    path = output_dir / f"checkpoint_epoch{epoch:03d}.pt"
    torch.save(ckpt, path)
    if best_so_far:
        torch.save(ckpt, output_dir / "best.pt")


def resume_if_available(model: nn.Module, optimizer: torch.optim.Optimizer, resume_path: Optional[Path]) -> int:
    if resume_path is None:
        return 0
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    checkpoint = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    print(f"[info] Resumed from {resume_path} at epoch {start_epoch}")
    return start_epoch


def filter_entries(
    entries: Iterable[Dict[str, str]],
    split: Optional[str],
    expert: Optional[str],
    limit: int,
) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []
    for entry in entries:
        if split and entry.get("split") != split:
            continue
        if expert and entry.get("expert") != expert:
            continue
        filtered.append(entry)
    if limit > 0:
        filtered = filtered[:limit]
    if not filtered:
        raise ValueError(f"No entries found for split={split!r}, expert={expert!r}")
    return filtered


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    entries = load_manifest(args.manifest)
    train_entries = filter_entries(entries, args.train_split, args.expert, args.train_limit)
    val_entries: Optional[List[Dict[str, str]]] = None
    if args.val_split:
        try:
            val_entries = filter_entries(entries, args.val_split, args.expert, args.val_limit)
        except ValueError:
            val_entries = None

    train_dataset = PreviewParamsDataset(
        train_entries,
        image_size=args.image_size,
        augment=not args.no_augment,
        imagenet_norm=args.imagenet_norm,
        load_raw_images=args.use_edit_layer,
        edit_resolution=args.edit_resolution,
    )
    val_dataset = None
    if val_entries:
        val_dataset = PreviewParamsDataset(
            val_entries,
            image_size=args.image_size,
            augment=False,
            imagenet_norm=args.imagenet_norm,
            load_raw_images=args.use_edit_layer,
            edit_resolution=args.edit_resolution,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
        )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initialize edit layer if requested
    edit_layer = None
    if args.use_edit_layer:
        edit_layer = DifferentiableEditLayer(tone_curve_resolution=1024).to(device)
        print(f"[info] Using DifferentiableEditLayer with image loss weight={args.image_loss_weight:.2f}")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = resume_if_available(model, optimizer, args.resume)
    best_val = math.inf

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            tone_weight=args.tone_weight,
            clip_grad=args.clip_grad,
            epoch=epoch,
            use_amp=args.amp and device.type == "cuda",
            edit_layer=edit_layer,
            image_loss_weight=args.image_loss_weight,
        )
        val_stats = None
        if val_loader is not None:
            val_stats = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                tone_weight=args.tone_weight,
                epoch=epoch,
                use_amp=args.amp and device.type == "cuda",
                edit_layer=edit_layer,
                image_loss_weight=args.image_loss_weight,
            )
            val_loss = val_stats["loss"]
            improvement = val_loss < best_val
            if improvement:
                best_val = val_loss
        else:
            improvement = True

        save_checkpoint(
            output_dir=args.output_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            stats=train_stats,
            val_stats=val_stats,
            best_so_far=improvement,
        )

        if val_stats:
            msg = f"[epoch {epoch}] train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f}"
            if train_stats.get('image_loss', 0) > 0:
                msg += f" train_img={train_stats['image_loss']:.4f} val_img={val_stats['image_loss']:.4f}"
            print(msg)
        else:
            msg = f"[epoch {epoch}] train_loss={train_stats['loss']:.4f}"
            if train_stats.get('image_loss', 0) > 0:
                msg += f" train_img={train_stats['image_loss']:.4f}"
            print(msg)


if __name__ == "__main__":
    main()

