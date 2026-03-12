"""
Test-Time Training (TTT) for VGGT.

This script implements test-time training using LoRA adapters on VGGT's
alternating attention modules. Inspired by Test3R's approach but adapted
for VGGT's architecture.

Usage:
    python train_ttt.py --model_name facebook/vggt-1b --dataset eth3d --epochs 2
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vggt.distillation.models import VGGTStudentModel
from vggt.distillation.dataset import VGGTDistillDataset
from vggt.training.ttt_loss import (
    ttt_consistency_loss,
    extract_anchor_predictions,
    make_triplet_pairs,
    TTTLoss,
)


def freeze_except_lora(model: nn.Module) -> int:
    """
    Freeze all parameters except LoRA adapters.

    Args:
        model: VGGTStudentModel with LoRA adapters

    Returns:
        Number of trainable parameters
    """
    trainable_params = 0

    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    return trainable_params


def inference_ttt_vggt(
    model: VGGTStudentModel,
    images: torch.Tensor,
    device: torch.device,
    epochs: int = 2,
    lr: float = 1e-5,
    accum_iter: int = 4,
    max_triplets: Optional[int] = None,
    use_confidence: bool = False,
    depth_weight: float = 0.1,
    use_amp: bool = True,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Test-time training for VGGT using LoRA adapters.

    Args:
        model: VGGTStudentModel with LoRA adapters
        images: [S, 3, H, W] sequence of images from test scene
        device: Device to run on
        epochs: Number of TTT epochs
        lr: Learning rate for TTT
        accum_iter: Gradient accumulation steps
        max_triplets: Maximum number of triplets per epoch (None = all)
        use_confidence: Whether to use confidence weighting in loss
        depth_weight: Weight for depth consistency loss
        use_amp: Whether to use automatic mixed precision
        verbose: Whether to print progress

    Returns:
        Dictionary with training statistics
    """
    if verbose:
        print(f"Starting TTT with {len(images)} images")

    # Set model to training mode
    model.train()

    # Freeze everything except LoRA
    trainable_params = freeze_except_lora(model)
    if verbose:
        print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    # Create loss function
    criterion = TTTLoss(use_confidence=use_confidence, depth_weight=depth_weight)

    # AMP scaler
    scaler = GradScaler(enabled=use_amp)

    # Create triplet pairs
    num_images = len(images)
    triplets = make_triplet_pairs(num_images, max_triplets=max_triplets)

    if verbose:
        print(f"Generated {len(triplets)} triplet pairs")

    # Training statistics
    stats = {
        'loss_per_epoch': [],
        'loss_per_step': [],
    }

    # Training loop
    for epoch in range(epochs):
        epoch_losses = []
        optimizer.zero_grad()

        pbar = tqdm(triplets, desc=f"TTT Epoch {epoch+1}/{epochs}") if verbose else triplets

        for step, (anchor_idx, j_idx, k_idx) in enumerate(pbar):
            # Prepare pair 1: (anchor, img_j)
            pair1_images = torch.stack([
                images[anchor_idx],
                images[j_idx]
            ]).unsqueeze(0).to(device)  # [1, 2, 3, H, W]

            # Prepare pair 2: (anchor, img_k)
            pair2_images = torch.stack([
                images[anchor_idx],
                images[k_idx]
            ]).unsqueeze(0).to(device)  # [1, 2, 3, H, W]

            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Get predictions for both pairs
                _, pred1 = model.forward_with_preds(pair1_images)
                _, pred2 = model.forward_with_preds(pair2_images)

                # Extract anchor predictions (index 0 in sequence)
                anchor_pred1 = extract_anchor_predictions(pred1, anchor_idx=0)
                anchor_pred2 = extract_anchor_predictions(pred2, anchor_idx=0)

                # Compute consistency loss
                loss = criterion(anchor_pred1, anchor_pred2)
                loss = loss / accum_iter

            # Backward pass
            scaler.scale(loss).backward()

            # Optimizer step with gradient accumulation
            if (step + 1) % accum_iter == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Record loss
            loss_value = loss.item() * accum_iter
            epoch_losses.append(loss_value)
            stats['loss_per_step'].append(loss_value)

            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f"{loss_value:.4f}"})

        # Final optimizer step if needed
        if len(triplets) % accum_iter != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Epoch statistics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        stats['loss_per_epoch'].append(avg_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    return stats


def train_ttt_on_scene(
    model: VGGTStudentModel,
    scene_name: str,
    dataset: VGGTDistillDataset,
    device: torch.device,
    epochs: int = 2,
    lr: float = 1e-5,
    save_path: Optional[str] = None,
    **ttt_kwargs,
) -> Dict[str, List[float]]:
    """
    Run TTT on a single scene.

    Args:
        model: VGGTStudentModel with LoRA adapters
        scene_name: Name of the scene
        dataset: Dataset to load images from
        device: Device to run on
        epochs: Number of TTT epochs
        lr: Learning rate
        save_path: Path to save LoRA weights (optional)
        **ttt_kwargs: Additional arguments for inference_ttt_vggt

    Returns:
        Training statistics
    """
    print(f"\n{'='*60}")
    print(f"TTT on scene: {scene_name}")
    print(f"{'='*60}")

    # Load scene data
    scene_data = dataset._load_scene_data(scene_name)
    image_files = scene_data['image_files']

    # Use fixed subset if configured
    if dataset.fixed_subset_seed is not None:
        subset_indices = dataset._get_fixed_subset_indices(scene_name, len(image_files))
        image_files = [image_files[i] for i in subset_indices]
        print(f"Using fixed subset: {len(image_files)} images")

    # Load images
    images = []
    for img_path in tqdm(image_files, desc="Loading images"):
        img = dataset._load_and_preprocess_image(img_path)
        images.append(torch.from_numpy(img).permute(2, 0, 1))  # [3, H, W]

    images = torch.stack(images)  # [S, 3, H, W]
    print(f"Loaded {len(images)} images with shape {images.shape}")

    # Run TTT
    stats = inference_ttt_vggt(
        model=model,
        images=images,
        device=device,
        epochs=epochs,
        lr=lr,
        **ttt_kwargs,
    )

    # Save LoRA weights if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_lora_weights(save_path)
        print(f"Saved LoRA weights to {save_path}")

    return stats


def train_ttt_on_dataset(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    epochs: int = 2,
    lr: float = 1e-5,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    device: str = "cuda",
    fixed_subset_seed: Optional[int] = 42,
    fixed_subset_max_frames: int = 100,
    max_triplets: Optional[int] = None,
    use_confidence: bool = False,
    depth_weight: float = 0.1,
    use_amp: bool = True,
    save_per_scene: bool = False,
):
    """
    Run TTT on all scenes in a dataset.

    Args:
        model_name: HuggingFace model name or path
        dataset_name: Name of dataset ('eth3d', '7scenes', etc.)
        output_dir: Directory to save checkpoints
        epochs: Number of TTT epochs per scene
        lr: Learning rate
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        device: Device to run on
        fixed_subset_seed: Seed for fixed frame subset (None = use all)
        fixed_subset_max_frames: Max frames in fixed subset
        max_triplets: Max triplets per epoch (None = all)
        use_confidence: Use confidence weighting in loss
        depth_weight: Weight for depth consistency
        use_amp: Use automatic mixed precision
        save_per_scene: Save separate checkpoint per scene
    """
    device = torch.device(device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {model_name}")
    model = VGGTStudentModel(
        model_name=model_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        train_camera_token=False,  # Don't train camera token for TTT
    ).to(device)

    # Create dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = VGGTDistillDataset(
        dataset_name=dataset_name,
        num_views=8,
        fixed_subset_seed=fixed_subset_seed,
        fixed_subset_max_frames=fixed_subset_max_frames,
        augment=False,  # No augmentation for TTT
    )

    # Get all scenes
    scenes = dataset.scenes
    print(f"Found {len(scenes)} scenes")

    # Train on each scene
    all_stats = {}

    for scene_idx, scene in enumerate(scenes):
        print(f"\n[{scene_idx+1}/{len(scenes)}] Processing scene: {scene}")

        # Determine save path
        if save_per_scene:
            save_path = os.path.join(output_dir, f"{scene}.pt")
        else:
            save_path = None

        # Run TTT
        try:
            stats = train_ttt_on_scene(
                model=model,
                scene_name=scene,
                dataset=dataset,
                device=device,
                epochs=epochs,
                lr=lr,
                save_path=save_path,
                max_triplets=max_triplets,
                use_confidence=use_confidence,
                depth_weight=depth_weight,
                use_amp=use_amp,
            )
            all_stats[scene] = stats

        except Exception as e:
            print(f"Error processing scene {scene}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save final averaged checkpoint
    if not save_per_scene:
        final_path = os.path.join(output_dir, "final.pt")
        model.save_lora_weights(final_path)
        print(f"\nSaved final averaged LoRA weights to {final_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("TTT Summary")
    print(f"{'='*60}")
    for scene, stats in all_stats.items():
        final_loss = stats['loss_per_epoch'][-1] if stats['loss_per_epoch'] else 0
        print(f"{scene}: Final loss = {final_loss:.4f}")

    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Test-Time Training for VGGT")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/vggt-1b",
                        help="HuggingFace model name or path")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="eth3d",
                        choices=['eth3d', '7scenes', 'scannetpp', 'hiroom', 'dtu'],
                        help="Dataset name")
    parser.add_argument("--fixed_subset_seed", type=int, default=42,
                        help="Seed for fixed frame subset (None = use all)")
    parser.add_argument("--fixed_subset_max_frames", type=int, default=100,
                        help="Max frames in fixed subset")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of TTT epochs per scene")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--accum_iter", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_triplets", type=int, default=None,
                        help="Max triplets per epoch (None = all)")

    # Loss arguments
    parser.add_argument("--use_confidence", action="store_true",
                        help="Use confidence weighting in loss")
    parser.add_argument("--depth_weight", type=float, default=0.1,
                        help="Weight for depth consistency loss")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/da3/checkpoints/vggt_ttt_lora",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_per_scene", action="store_true",
                        help="Save separate checkpoint per scene")

    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")

    args = parser.parse_args()

    # Run TTT
    train_ttt_on_dataset(
        model_name=args.model_name,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=args.device,
        fixed_subset_seed=args.fixed_subset_seed,
        fixed_subset_max_frames=args.fixed_subset_max_frames,
        max_triplets=args.max_triplets,
        use_confidence=args.use_confidence,
        depth_weight=args.depth_weight,
        use_amp=not args.no_amp,
        save_per_scene=args.save_per_scene,
    )


if __name__ == "__main__":
    main()
