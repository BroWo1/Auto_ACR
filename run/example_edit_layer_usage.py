#!/usr/bin/env python3
"""Example: How to use DifferentiableEditLayer for end-to-end training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path

from model.autopilot import AutoPilotRegressor, BackboneConfig
from model.edit_layer import DifferentiableEditLayer
from params.ranges import SCALAR_NAMES


def example_1_basic_usage():
    """Example 1: Basic usage of DifferentiableEditLayer."""

    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create the edit layer
    edit_layer = DifferentiableEditLayer(tone_curve_resolution=1024)

    # Simulate a batch of images (linear RGB, [0,1] range)
    batch_size = 2
    images = torch.rand(batch_size, 3, 512, 512)  # Random images

    # Create normalized parameters (what the model predicts)
    params_norm = {
        'Temperature': torch.tensor([0.2, -0.3]),      # [-1, 1]
        'Tint': torch.tensor([0.0, 0.1]),
        'Exposure2012': torch.tensor([0.5, -0.2]),
        'Contrast2012': torch.tensor([0.3, 0.1]),
        'Highlights2012': torch.tensor([-0.2, 0.0]),
        'Shadows2012': torch.tensor([0.4, 0.3]),
        'Whites2012': torch.tensor([0.1, -0.1]),
        'Blacks2012': torch.tensor([-0.3, 0.0]),
        'Vibrance': torch.tensor([0.2, 0.4]),
        'Saturation': torch.tensor([0.1, 0.2]),
    }

    # Tone curve: 16 knots in [0, 1], monotonically increasing
    tone_curve = torch.linspace(0, 1, 16).unsqueeze(0).expand(batch_size, -1)

    # Apply edits
    edited_images = edit_layer(images, params_norm, tone_curve)

    print(f"Input shape:  {images.shape}")
    print(f"Output shape: {edited_images.shape}")
    print(f"Output range: [{edited_images.min():.3f}, {edited_images.max():.3f}]")
    print()


def example_2_with_model():
    """Example 2: Use DifferentiableEditLayer with AutoPilotRegressor."""

    print("=" * 60)
    print("Example 2: End-to-End Model + Edit Layer")
    print("=" * 60)

    # Create model
    cfg = BackboneConfig(name='vit_small_patch16_224', pretrained=False)
    model = AutoPilotRegressor(backbone=cfg, metadata_encoder=None)
    edit_layer = DifferentiableEditLayer(tone_curve_resolution=1024)

    # Simulate input
    batch_size = 4
    input_images = torch.rand(batch_size, 3, 224, 224)  # sRGB previews

    # Forward pass through model
    output = model(input_images)
    params_norm = output['params_norm']

    # Extract tone curve
    tone_curve = params_norm.pop('ToneCurve')  # [batch, 16]

    print(f"Model predicted parameters:")
    for name, value in params_norm.items():
        print(f"  {name:20s}: {value.shape} range [{value.min():.3f}, {value.max():.3f}]")
    print(f"  {'ToneCurve':20s}: {tone_curve.shape} range [{tone_curve.min():.3f}, {tone_curve.max():.3f}]")

    # Apply edits (upsample input for editing)
    # In practice, you'd use the original high-res linear ProPhoto RGB image
    upsampled = F.interpolate(input_images, size=(512, 512), mode='bilinear')
    edited_images = edit_layer(upsampled, params_norm, tone_curve)

    print(f"\nEdited image shape: {edited_images.shape}")
    print()


def example_3_training_with_image_loss():
    """Example 3: Training loop using image-space loss."""

    print("=" * 60)
    print("Example 3: Training with Image-Space Loss")
    print("=" * 60)

    # Setup
    cfg = BackboneConfig(name='vit_small_patch16_224', pretrained=False)
    model = AutoPilotRegressor(backbone=cfg, metadata_encoder=None)
    edit_layer = DifferentiableEditLayer(tone_curve_resolution=1024)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Simulate a training batch
    batch_size = 2
    preview_images = torch.rand(batch_size, 3, 224, 224)  # Model input (sRGB preview)
    raw_images = torch.rand(batch_size, 3, 512, 512)      # High-res linear RGB
    target_images = torch.rand(batch_size, 3, 512, 512)   # Target edited images

    # Training step
    optimizer.zero_grad()

    # 1. Model predicts parameters from preview
    output = model(preview_images)
    params_norm = output['params_norm']
    tone_curve = params_norm.pop('ToneCurve')

    # 2. Apply predicted parameters to raw image
    edited_images = edit_layer(raw_images, params_norm, tone_curve)

    # 3. Compute loss in IMAGE space (not parameter space!)
    image_loss = F.mse_loss(edited_images, target_images)

    # 4. Optional: Add parameter regularization
    param_reg = sum(p.pow(2).mean() for p in params_norm.values()) * 0.01
    tone_reg = (tone_curve[:, 1:] - tone_curve[:, :-1]).pow(2).mean() * 0.001

    total_loss = image_loss + param_reg + tone_reg

    print(f"Image loss:      {image_loss.item():.6f}")
    print(f"Param reg:       {param_reg.item():.6f}")
    print(f"Tone reg:        {tone_reg.item():.6f}")
    print(f"Total loss:      {total_loss.item():.6f}")

    # Backward pass
    total_loss.backward()
    optimizer.step()

    print("✓ Gradients computed and optimizer stepped")
    print()


def example_4_hybrid_loss():
    """Example 4: Hybrid loss - both parameter MSE and image MSE."""

    print("=" * 60)
    print("Example 4: Hybrid Parameter + Image Loss")
    print("=" * 60)

    cfg = BackboneConfig(name='vit_small_patch16_224', pretrained=False)
    model = AutoPilotRegressor(backbone=cfg, metadata_encoder=None)
    edit_layer = DifferentiableEditLayer(tone_curve_resolution=1024)

    # Simulate data
    batch_size = 2
    preview_images = torch.rand(batch_size, 3, 224, 224)
    raw_images = torch.rand(batch_size, 3, 512, 512)
    target_images = torch.rand(batch_size, 3, 512, 512)

    # Ground truth parameters (from fitting)
    gt_scalars = {name: torch.randn(batch_size) for name in SCALAR_NAMES}
    gt_tone = torch.linspace(0, 1, 16).unsqueeze(0).expand(batch_size, -1)

    # Predict
    output = model(preview_images)
    pred_params = output['params_norm']
    pred_tone = pred_params.pop('ToneCurve')

    # Loss 1: Parameter space (like current training)
    scalar_loss = sum(
        F.mse_loss(pred_params[name], gt_scalars[name])
        for name in SCALAR_NAMES
    ) / len(SCALAR_NAMES)
    tone_loss = F.mse_loss(pred_tone, gt_tone)
    param_loss = scalar_loss + tone_loss

    # Loss 2: Image space (apply edits and compare)
    pred_edited = edit_layer(raw_images, pred_params, pred_tone)
    image_loss = F.mse_loss(pred_edited, target_images)

    # Combined loss
    alpha = 0.7  # Weight for image loss
    beta = 0.3   # Weight for parameter loss
    total_loss = alpha * image_loss + beta * param_loss

    print(f"Parameter loss:  {param_loss.item():.6f}")
    print(f"Image loss:      {image_loss.item():.6f}")
    print(f"Total loss:      {total_loss.item():.6f}")
    print()
    print("Hybrid approach ensures:")
    print("  - Model learns correct parameters (param_loss)")
    print("  - Edits produce visually correct results (image_loss)")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DifferentiableEditLayer Usage Examples")
    print("=" * 60 + "\n")

    example_1_basic_usage()
    example_2_with_model()
    example_3_training_with_image_loss()
    example_4_hybrid_loss()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The DifferentiableEditLayer allows you to:

1. Apply ACR-style edits differentiably
2. Backpropagate through the editing pipeline
3. Train with image-space loss instead of just parameter MSE
4. Use hybrid losses (params + images)

Key advantages:
✓ Model learns to produce good-looking images, not just fit parameters
✓ Gradients flow from final image back to model
✓ Can add perceptual losses, adversarial losses, etc.
✓ More robust to fitting errors in ground truth parameters

To integrate into training:
1. Add edit_layer = DifferentiableEditLayer() to your training script
2. Load both preview (for model) and raw (for editing)
3. Compute image_loss after applying edits
4. Combine with param_loss for stability
    """)