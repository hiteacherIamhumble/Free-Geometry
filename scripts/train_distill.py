#!/usr/bin/env python3
"""
Training Script for DA3 Knowledge Distillation.

This script trains a student model (4 views + LoRA) to match a teacher model
(8 views, frozen DA3-Giant) using feature distillation.

Usage:
    python scripts/train_distill.py --data_root ./data --output_dir ./checkpoints/distill

    # Debug mode (small batch, few steps)
    python scripts/train_distill.py --data_root ./data --debug

    # Resume from checkpoint
    python scripts/train_distill.py --data_root ./data --resume ./checkpoints/distill/latest.pt
"""

import argparse
import os
import random
import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_anything_3.distillation.config import DistillConfig, get_default_config, get_debug_config
from depth_anything_3.distillation.dataset import ScanNetPPDistillDataset
from depth_anything_3.distillation.benchmark_dataset import BenchmarkDistillDataset
from depth_anything_3.distillation.multi_dataset import create_multi_dataset
from depth_anything_3.distillation.losses import (
    CombinedTokenSoftmaxKLCosineLoss,
    AllTokenSoftmaxKLCosineLoss,
)
from depth_anything_3.distillation.models import TeacherModel, StudentModel


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    warmup_steps: int,
    steps_per_epoch: int | None = None,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    if scheduler_type in (None, "none"):
        return None
    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = None
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=1e-6,
        )
        if warmup_scheduler is not None:
            return SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            return cosine_scheduler
    elif scheduler_type == "step":
        from torch.optim.lr_scheduler import StepLR
        step_size = max(1, steps_per_epoch or num_training_steps)
        return StepLR(optimizer, step_size=step_size, gamma=0.7)
    elif scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps,
        )
    elif scheduler_type == "constant":
        from torch.optim.lr_scheduler import ConstantLR
        return ConstantLR(optimizer, factor=1.0, total_iters=num_training_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def save_checkpoint(
    student: StudentModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    loss: float,
    output_dir: str,
    filename: str = "checkpoint.pt",
) -> str:
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'loss': loss,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict(),
    }

    # Save LoRA weights separately
    lora_path = os.path.join(output_dir, filename.replace('.pt', '_lora.pt'))
    student.save_lora_weights(lora_path)
    checkpoint['lora_path'] = lora_path

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Also save as latest
    latest_path = os.path.join(output_dir, "latest.pt")
    latest_lora_path = os.path.join(output_dir, "latest_lora.pt")
    torch.save(checkpoint, latest_path)
    student.save_lora_weights(latest_lora_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    student: StudentModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
) -> Dict:
    """Load training checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load LoRA weights
    if 'lora_path' in checkpoint and os.path.exists(checkpoint['lora_path']):
        student.load_lora_weights(checkpoint['lora_path'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint


def train_epoch(
    teacher: TeacherModel,
    student: StudentModel,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    config: DistillConfig,
    writer: Optional[SummaryWriter] = None,
) -> int:
    """Train for one epoch."""
    student.train()
    teacher.eval()

    device = config.training.device
    log_interval = config.training.log_interval
    save_interval = config.training.save_interval
    grad_accum_steps = config.training.gradient_accumulation_steps

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        teacher_images = batch['teacher_images'].to(device)  # [B, 8, 3, H, W]
        student_images = batch['student_images'].to(device)  # [B, 4, 3, H, W]

        # Forward pass
        with autocast(enabled=config.training.use_amp):
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_output = teacher(teacher_images)

            # Student forward
            student_output = student(student_images)

            # Compute loss
            loss, loss_details = criterion(teacher_output, student_output)
            loss = loss / grad_accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                student.get_trainable_params(),
                config.training.max_grad_norm,
            )

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            global_step += 1

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            if scheduler is not None:
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * config.data.batch_size / elapsed

            # Print main metrics
            print(
                f"Epoch {epoch} | Step {global_step} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                f"Speed: {samples_per_sec:.1f} samples/s"
            )

            # Print detailed loss breakdown
            if loss_details:
                detail_str = " | ".join([
                    f"{k}: {v:.4f}" for k, v in loss_details.items()
                    if k not in ['total_loss', 'num_layers']  # Skip redundant keys
                ])
                print(f"  Loss details: {detail_str}")

            if writer is not None:
                writer.add_scalar('train/loss', avg_loss, global_step)
                writer.add_scalar('train/lr', lr, global_step)
                for key, value in loss_details.items():
                    writer.add_scalar(f'train/{key}', value, global_step)

        # Save checkpoint
        if save_interval > 0 and global_step % save_interval == 0:
            save_checkpoint(
                student, optimizer, scheduler, scaler,
                epoch, global_step, total_loss / num_batches,
                config.training.output_dir,
                f"checkpoint_step{global_step}.pt",
            )

    return global_step


@torch.no_grad()
def evaluate(
    teacher: TeacherModel,
    student: StudentModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: DistillConfig,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    student.eval()
    teacher.eval()

    device = config.training.device
    total_loss = 0.0
    total_details = {}
    num_batches = 0

    for batch in val_loader:
        teacher_images = batch['teacher_images'].to(device)
        student_images = batch['student_images'].to(device)

        with autocast(enabled=config.training.use_amp):
            teacher_output = teacher(teacher_images)
            student_output = student(student_images)
            loss, loss_details = criterion(teacher_output, student_output)

        total_loss += loss.item()
        for key, value in loss_details.items():
            total_details[key] = total_details.get(key, 0.0) + value
        num_batches += 1

    # Average
    avg_loss = total_loss / num_batches
    avg_details = {k: v / num_batches for k, v in total_details.items()}
    avg_details['loss'] = avg_loss

    return avg_details


def main():
    parser = argparse.ArgumentParser(description='DA3 Knowledge Distillation Training')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='scannetpp',
                        choices=['scannetpp', 'eth3d', '7scenes', 'hiroom', 'dtu'],
                        help='Dataset to use for training')
    parser.add_argument('--use_multi_dataset', action='store_true',
                        help='Use multi-dataset (ScanNet++ train + all benchmark datasets)')
    parser.add_argument('--samples_per_scene', type=int, default=4,
                        help='Number of samples per scene for benchmark datasets (default: 4)')
    parser.add_argument('--seeds_list', type=int, nargs='+', default=None,
                        help='Optional list of seeds for sampling benchmark datasets (overrides samples_per_scene)')
    parser.add_argument('--first_frame_ref', action='store_true',
                        help='Force first frame as reference after sampling (sorted indices, smallest first)')
    parser.add_argument('--eth3d_samples', type=int, default=None,
                        help='Number of samples per scene for ETH3D (overrides --samples_per_scene)')
    parser.add_argument('--7scenes_samples', type=int, default=None,
                        help='Number of samples per scene for 7Scenes (overrides --samples_per_scene)')
    parser.add_argument('--use_scannetpp_test_split', action='store_true',
                        help='Use benchmark_dataset/scannetpp (test split) instead of ScanNetPPDistillDataset')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num_views', type=int, default=8,
                        help='Number of views for teacher (8 or 16)')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Warmup steps for LR scheduler (override config default 500)')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='depth-anything/DA3-GIANT-1.1',
                        help='HuggingFace model name')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=16.0,
                        help='LoRA alpha')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=['cosine', 'linear', 'constant', 'step', 'none'],
                        help='LR scheduler type (override config default)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay (override config default)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/distill',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--early_stopping_patience', type=int, default=0,
                        help='Early stopping patience (0 to disable)')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (small batch, few steps)')
    parser.add_argument('--ref_view_strategy', type=str, default='first',
                        help='Reference view strategy (first, saddle_balanced, etc.)')

    # Loss function arguments
    parser.add_argument('--all_token_softmax_kl_cosine', action='store_true',
                        help='Full 3072-dim token softmax KL + cosine loss (no local/global split)')
    parser.add_argument('--all_token_softmax_kl_weight', type=float, default=1.0,
                        help='Weight for full-token softmax KL term')
    parser.add_argument('--all_token_softmax_cos_weight', type=float, default=1.0,
                        help='Weight for full-token softmax cosine term')
    parser.add_argument('--combined_token_softmax_kl_cosine', action='store_true',
                        help='Use combined local+global per-token softmax KL + cosine (sum of both)')
    parser.add_argument('--local_token_softmax_kl_weight', type=float, default=1.0,
                        help='Weight for local softmax KL term')
    parser.add_argument('--local_token_softmax_cos_weight', type=float, default=1.0,
                        help='Weight for local softmax cosine term')
    parser.add_argument('--global_token_softmax_kl_weight', type=float, default=1.0,
                        help='Weight for global softmax KL term')
    parser.add_argument('--global_token_softmax_cos_weight', type=float, default=1.0,
                        help='Weight for global softmax cosine term')

    args = parser.parse_args()

    # Get configuration
    if args.debug:
        config = get_debug_config()
    else:
        config = get_default_config()

    # Override with command line arguments
    config.data.data_root = args.data_root
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    config.data.num_views = args.num_views
    # Set student indices based on num_views
    if args.num_views == 16:
        config.data.student_indices = [0, 4, 8, 12]  # Every 4th frame
    else:  # 8 views (default)
        config.data.student_indices = [0, 2, 4, 6]  # Every 2nd frame
    config.model.model_name = args.model_name
    config.model.lora_rank = args.lora_rank
    config.model.lora_alpha = args.lora_alpha
    config.training.epochs = args.epochs
    config.training.lr = args.lr
    if args.warmup_steps is not None:
        config.training.warmup_steps = args.warmup_steps
    if args.lr_scheduler is not None:
        config.training.lr_scheduler = args.lr_scheduler
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    config.training.output_dir = args.output_dir
    config.training.seed = args.seed

    # Set seed
    set_seed(config.training.seed)

    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)

    # Setup device
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Teacher views: {config.data.num_views}")
    print(f"  Student views: 4 (indices: {config.data.student_indices})")

    # Create datasets
    print("\nCreating datasets...")

    # Choose dataset type based on argument
    if args.use_multi_dataset:
        # Use multi-dataset loader (ScanNet++ train + all benchmark datasets)
        print("Using multi-dataset mode (ScanNet++ train + all benchmark datasets)")

        # Build per-dataset samples dict if custom values specified
        per_dataset_samples = {}
        if args.eth3d_samples is not None:
            per_dataset_samples['eth3d'] = args.eth3d_samples
        if args.__dict__.get('7scenes_samples') is not None:
            per_dataset_samples['7scenes'] = args.__dict__['7scenes_samples']

        train_dataset = create_multi_dataset(
            scannetpp_data_root=config.data.data_root,
            benchmark_data_root='/home/22097845d/Depth-Anything-3/workspace/benchmark_dataset',
            num_views=config.data.num_views,
            student_indices=config.data.student_indices,
            augment=config.data.augment,
            samples_per_scene=args.samples_per_scene,
            seed=config.training.seed,
            image_size=config.data.image_size,
            per_dataset_samples=per_dataset_samples if per_dataset_samples else None,
        )
        # No validation dataset for multi-dataset mode
        val_dataset = None
    elif args.dataset == 'scannetpp' and not args.use_scannetpp_test_split:
        # Use original ScanNetPP dataset
        train_dataset = ScanNetPPDistillDataset(
            data_root=config.data.data_root,
            split='train',
            num_views=config.data.num_views,
            image_size=config.data.image_size,
            student_indices=config.data.student_indices,
            augment=config.data.augment,
        )
        val_dataset = ScanNetPPDistillDataset(
            data_root=config.data.data_root,
            split='val',
            num_views=config.data.num_views,
            image_size=config.data.image_size,
            student_indices=config.data.student_indices,
            augment=False,
        )
    elif args.dataset == 'scannetpp' and args.use_scannetpp_test_split:
        bench_samples = args.samples_per_scene
        if args.seeds_list is not None:
            bench_samples = len(args.seeds_list)
        train_dataset = BenchmarkDistillDataset(
            dataset_name='scannetpp',
            num_views=config.data.num_views,
            image_size=config.data.image_size,
            student_indices=config.data.student_indices,
            augment=config.data.augment,
            samples_per_scene=bench_samples,
            seed=config.training.seed,
            seeds_list=args.seeds_list,
            first_frame_ref=args.first_frame_ref,
        )
        val_dataset = None
    else:
        # Use benchmark dataset loader (all scenes for training, no val split)
        bench_samples = args.samples_per_scene
        if args.dataset == 'eth3d' and args.eth3d_samples is not None:
            bench_samples = args.eth3d_samples
        if args.seeds_list is not None:
            bench_samples = len(args.seeds_list)
        train_dataset = BenchmarkDistillDataset(
            dataset_name=args.dataset,
            num_views=config.data.num_views,
            image_size=config.data.image_size,
            student_indices=config.data.student_indices,
            augment=config.data.augment,
            samples_per_scene=bench_samples,
            seed=config.training.seed,
            seeds_list=args.seeds_list,
            first_frame_ref=args.first_frame_ref,
        )
        # No validation dataset for benchmark datasets
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create validation loader only if val_dataset exists
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True,
        )
    else:
        val_loader = None

    print(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val samples: {len(val_dataset)}")
    else:
        print("No validation set (using all scenes for training)")


    # Create models
    print("\nCreating models...")
    teacher = TeacherModel(
        model_name=config.model.model_name,
        output_layers=config.model.output_layers,
        embed_dim=config.model.embed_dim,
        ref_view_strategy=args.ref_view_strategy,
    ).to(device)

    student = StudentModel(
        model_name=config.model.model_name,
        output_layers=config.model.output_layers,
        embed_dim=config.model.embed_dim,
        lora_rank=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        train_camera_token=config.model.train_camera_token,
        ref_view_strategy=args.ref_view_strategy,
    ).to(device)

    # Create loss function
    if args.all_token_softmax_kl_cosine:
        print("\nUsing AllTokenSoftmaxKLCosineLoss (full 3072-dim softmax KL + cosine)")
        print("  - Operates on concatenated token (no local/global split)")
        criterion = AllTokenSoftmaxKLCosineLoss(
            student_frame_indices=config.data.student_indices,
            kl_weight=args.all_token_softmax_kl_weight,
            cos_weight=args.all_token_softmax_cos_weight,
        )
    elif args.combined_token_softmax_kl_cosine:
        print("\nUsing CombinedTokenSoftmaxKLCosineLoss (local+global per-token softmax KL + cosine)")
        criterion = CombinedTokenSoftmaxKLCosineLoss(
            student_frame_indices=config.data.student_indices,
            local_kl_weight=args.local_token_softmax_kl_weight,
            local_cos_weight=args.local_token_softmax_cos_weight,
            global_kl_weight=args.global_token_softmax_kl_weight,
            global_cos_weight=args.global_token_softmax_cos_weight,
        )
    else:
        # Default to combined_token_softmax_kl_cosine
        print("\nUsing CombinedTokenSoftmaxKLCosineLoss (default)")
        criterion = CombinedTokenSoftmaxKLCosineLoss(
            student_frame_indices=config.data.student_indices,
            local_kl_weight=args.local_token_softmax_kl_weight,
            local_cos_weight=args.local_token_softmax_cos_weight,
            global_kl_weight=args.global_token_softmax_kl_weight,
            global_cos_weight=args.global_token_softmax_cos_weight,
        )

    # Create optimizer (only LoRA params)
    optimizer = torch.optim.AdamW(
        student.get_trainable_params(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # Create scheduler
    num_training_steps = len(train_loader) * config.training.epochs
    scheduler = get_lr_scheduler(
        optimizer,
        config.training.lr_scheduler,
        num_training_steps,
        config.training.warmup_steps,
        len(train_loader),
    )

    # Create scaler for mixed precision
    scaler = GradScaler(enabled=config.training.use_amp)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, student, optimizer, scheduler, scaler)
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']

    # Create tensorboard writer
    log_dir = os.path.join(config.training.output_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(start_epoch, config.training.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.training.epochs}")
        print(f"{'='*60}")

        # Train
        global_step = train_epoch(
            teacher, student, train_loader, criterion,
            optimizer, scheduler, scaler,
            epoch, global_step, config, writer,
        )

        # Evaluate
        if config.training.eval_interval > 0 and val_loader is not None:
            print("\nEvaluating...")
            val_metrics = evaluate(teacher, student, val_loader, criterion, config)
            print(f"Validation loss: {val_metrics['loss']:.4f}")

            # Log to tensorboard
            for key, value in val_metrics.items():
                writer.add_scalar(f'val/{key}', value, global_step)

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                epochs_without_improvement = 0
                save_checkpoint(
                    student, optimizer, scheduler, scaler,
                    epoch, global_step, val_metrics['loss'],
                    config.training.output_dir,
                    "best.pt",
                )
            else:
                epochs_without_improvement += 1

            # Early stopping check
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement")
                break

        # Save epoch checkpoint
        save_checkpoint(
            student, optimizer, scheduler, scaler,
            epoch, global_step, 0.0,
            config.training.output_dir,
            f"epoch_{epoch}.pt",
        )

    print("\nTraining complete!")
    writer.close()


if __name__ == '__main__':
    main()
