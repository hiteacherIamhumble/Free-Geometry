#!/usr/bin/env python3
"""
Training Script for VGGT Knowledge Distillation.

This script trains a student model (4 views + LoRA) to match a teacher model
(8 views, frozen VGGT) using feature distillation.

Usage:
    python scripts/train_distill_vggt.py --data_root ./data --output_dir ./checkpoints/vggt_distill

    # Debug mode (small batch, few steps)
    python scripts/train_distill_vggt.py --data_root ./data --debug

    # Resume from checkpoint
    python scripts/train_distill_vggt.py --data_root ./data --resume ./checkpoints/vggt_distill/latest.pt
"""

import argparse
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Add vggt submodule to path (for internal vggt imports like 'from vggt.models.vggt import VGGT')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'vggt'))

from vggt.vggt.distillation.models import VGGTTeacherModel, VGGTStudentModel, VGGTDistillationOutput
from vggt.vggt.distillation.dataset import VGGTDistillDataset


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


class VGGTCombinedSoftmaxKLCosineLoss(nn.Module):
    """
    Combined frame+global per-token softmax KL + cosine loss for VGGT.

    Adapts the DA3 loss pattern for VGGT's 2048-dim features (1024 frame + 1024 global).
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        target_layer: int = 23,
        frame_kl_weight: float = 1.0,
        frame_cos_weight: float = 1.0,
        global_kl_weight: float = 1.0,
        global_cos_weight: float = 1.0,
    ):
        super().__init__()
        self.target_layer = target_layer
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.frame_kl_weight = frame_kl_weight
        self.frame_cos_weight = frame_cos_weight
        self.global_kl_weight = global_kl_weight
        self.global_cos_weight = global_cos_weight
        print(f"VGGTCombinedSoftmaxKLCosineLoss: layer={target_layer}")
        print(f"  Frame: kl_w={frame_kl_weight}, cos_w={frame_cos_weight}")
        print(f"  Global: kl_w={global_kl_weight}, cos_w={global_cos_weight}")

    def forward(
        self,
        teacher_output: VGGTDistillationOutput,
        student_output: VGGTDistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Frame features loss
        teacher_frame = teacher_output.frame_features[self.target_layer]
        student_frame = student_output.frame_features[self.target_layer]
        teacher_frame_selected = teacher_frame[:, self.student_frame_indices, :, :]

        # Per-token channel softmax for frame
        teacher_frame_sm = torch.softmax(teacher_frame_selected, dim=-1)
        student_frame_sm = torch.softmax(student_frame, dim=-1)

        frame_kl = (teacher_frame_sm * (torch.log(teacher_frame_sm + 1e-8) - torch.log(student_frame_sm + 1e-8))).sum(dim=-1).mean()
        teacher_frame_norm = torch.nn.functional.normalize(teacher_frame_sm, dim=-1)
        student_frame_norm = torch.nn.functional.normalize(student_frame_sm, dim=-1)
        frame_cos = (teacher_frame_norm * student_frame_norm).sum(dim=-1).mean()

        frame_loss = self.frame_kl_weight * frame_kl + self.frame_cos_weight * (1.0 - frame_cos)

        # Global features loss
        teacher_global = teacher_output.global_features[self.target_layer]
        student_global = student_output.global_features[self.target_layer]
        teacher_global_selected = teacher_global[:, self.student_frame_indices, :, :]

        # Per-token channel softmax for global
        teacher_global_sm = torch.softmax(teacher_global_selected, dim=-1)
        student_global_sm = torch.softmax(student_global, dim=-1)

        global_kl = (teacher_global_sm * (torch.log(teacher_global_sm + 1e-8) - torch.log(student_global_sm + 1e-8))).sum(dim=-1).mean()
        teacher_global_norm = torch.nn.functional.normalize(teacher_global_sm, dim=-1)
        student_global_norm = torch.nn.functional.normalize(student_global_sm, dim=-1)
        global_cos = (teacher_global_norm * student_global_norm).sum(dim=-1).mean()

        global_loss = self.global_kl_weight * global_kl + self.global_cos_weight * (1.0 - global_cos)

        # Combined loss
        loss = frame_loss + global_loss

        return loss, {
            'frame_softmax_kl': frame_kl.item(),
            'frame_softmax_cos': frame_cos.item(),
            'frame_loss': frame_loss.item(),
            'global_softmax_kl': global_kl.item(),
            'global_softmax_cos': global_cos.item(),
            'global_loss': global_loss.item(),
            'total_loss': loss.item(),
        }


class VGGTAllTokenSoftmaxKLCosineLoss(nn.Module):
    """
    Full-token (2048-dim) channel softmax KL + cosine loss for VGGT.

    Treats the concatenated [frame, global] token as one vector.
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        target_layer: int = 23,
        kl_weight: float = 1.0,
        cos_weight: float = 1.0,
    ):
        super().__init__()
        self.target_layer = target_layer
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.kl_weight = kl_weight
        self.cos_weight = cos_weight
        print(f"VGGTAllTokenSoftmaxKLCosineLoss: layer={target_layer}, kl_w={kl_weight}, cos_w={cos_weight}")
        print("  Uses full 2048-dim token (no frame/global split)")

    def forward(
        self,
        teacher_output: VGGTDistillationOutput,
        student_output: VGGTDistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_all = teacher_output.layer_features[self.target_layer]  # [B, 8, P, 2048]
        student_all = student_output.layer_features[self.target_layer]  # [B, 4, P, 2048]
        teacher_selected = teacher_all[:, self.student_frame_indices, :, :]

        # Per-token channel softmax over the full 2048 dims
        teacher_sm = torch.softmax(teacher_selected, dim=-1)
        student_sm = torch.softmax(student_all, dim=-1)

        kl = (teacher_sm * (torch.log(teacher_sm + 1e-8) - torch.log(student_sm + 1e-8))).sum(dim=-1).mean()

        # Cosine on the same full token
        teacher_norm = torch.nn.functional.normalize(teacher_selected, dim=-1)
        student_norm = torch.nn.functional.normalize(student_all, dim=-1)
        cos = (teacher_norm * student_norm).sum(dim=-1).mean()

        loss = self.kl_weight * kl + self.cos_weight * (1.0 - cos)

        return loss, {
            'all_softmax_kl': kl.item(),
            'all_softmax_cos': cos.item(),
            'all_softmax_total': loss.item(),
            'total_loss': loss.item(),
        }


def save_checkpoint(
    student: VGGTStudentModel,
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
    student: VGGTStudentModel,
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
    teacher: VGGTTeacherModel,
    student: VGGTStudentModel,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    args,
    writer: Optional[SummaryWriter] = None,
) -> int:
    """Train for one epoch."""
    student.train()
    teacher.eval()

    device = args.device
    log_interval = args.log_interval
    save_interval = args.save_interval
    grad_accum_steps = args.gradient_accumulation_steps

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        teacher_images = batch['teacher_images'].to(device)  # [B, 8, 3, H, W]
        student_images = batch['student_images'].to(device)  # [B, 4, 3, H, W]

        # Forward pass
        with autocast(enabled=args.use_amp):
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
                args.max_grad_norm,
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
            samples_per_sec = (batch_idx + 1) * args.batch_size / elapsed

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
                    if k not in ['total_loss']
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
                args.output_dir,
                f"checkpoint_step{global_step}.pt",
            )

    return global_step


@torch.no_grad()
def evaluate(
    teacher: VGGTTeacherModel,
    student: VGGTStudentModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    args,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    student.eval()
    teacher.eval()

    device = args.device
    total_loss = 0.0
    total_details = {}
    num_batches = 0

    for batch in val_loader:
        teacher_images = batch['teacher_images'].to(device)
        student_images = batch['student_images'].to(device)

        with autocast(enabled=args.use_amp):
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
    parser = argparse.ArgumentParser(description='VGGT Knowledge Distillation Training')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='scannetpp',
                        choices=['scannetpp', 'eth3d', '7scenes', 'hiroom', 'dtu'],
                        help='Dataset to use for training')
    parser.add_argument('--samples_per_scene', type=int, default=4,
                        help='Number of samples per scene (default: 4)')
    parser.add_argument('--seeds_list', type=int, nargs='+', default=None,
                        help='Optional list of seeds for sampling')
    parser.add_argument('--first_frame_ref', action='store_true',
                        help='Force first frame as reference after sampling')
    parser.add_argument('--subset_sampling', action='store_true',
                        help='Enable subset-based frame sampling (recommended for large scenes)')
    parser.add_argument('--subset_ratio', type=float, default=0.05,
                        help='Ratio of frames to include in subset (default: 0.05 = 5%%)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num_views', type=int, default=8,
                        help='Number of views for teacher (8 or 16)')
    parser.add_argument('--student_views', type=int, default=None,
                        help='Number of student views (default: num_views/2)')
    parser.add_argument('--image_size', type=int, default=504,
                        help='Image size (longest side, DA3-style, default: 504)')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='facebook/vggt-1b',
                        help='HuggingFace model name')
    parser.add_argument('--output_layers', type=int, nargs='+', default=[19, 23],
                        help='Layers to extract features from')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=16.0,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout')
    parser.add_argument('--lora_layers_start', type=int, default=12,
                        help='First layer to apply LoRA (default: 12)')
    parser.add_argument('--train_camera_token', action='store_true', default=True,
                        help='Make camera token trainable')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='none',
                        choices=['cosine', 'linear', 'constant', 'step', 'none'],
                        help='LR scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup steps for LR scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/vggt_distill',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')

    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='Save checkpoint every N steps (0 to disable)')

    # Loss function arguments
    parser.add_argument('--all_token_softmax_kl_cosine', action='store_true',
                        help='Full 2048-dim token softmax KL + cosine loss')
    parser.add_argument('--all_token_kl_weight', type=float, default=1.0,
                        help='Weight for full-token softmax KL term')
    parser.add_argument('--all_token_cos_weight', type=float, default=2.0,
                        help='Weight for full-token softmax cosine term')
    parser.add_argument('--combined_token_softmax_kl_cosine', action='store_true',
                        help='Use combined frame+global per-token softmax KL + cosine')
    parser.add_argument('--frame_kl_weight', type=float, default=1.0,
                        help='Weight for frame softmax KL term')
    parser.add_argument('--frame_cos_weight', type=float, default=2.0,
                        help='Weight for frame softmax cosine term')
    parser.add_argument('--global_kl_weight', type=float, default=1.0,
                        help='Weight for global softmax KL term')
    parser.add_argument('--global_cos_weight', type=float, default=2.0,
                        help='Weight for global softmax cosine term')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (small batch, few steps)')

    args = parser.parse_args()

    # Debug mode overrides
    if args.debug:
        args.epochs = 1
        args.batch_size = 1
        args.samples_per_scene = 1
        args.log_interval = 1
        args.save_interval = 0

    # Determine student views
    if args.student_views is None:
        args.student_views = args.num_views // 2

    # Set student indices (even-indexed from teacher views)
    step = args.num_views // args.student_views
    student_indices = [i * step for i in range(args.student_views)]
    # For 16v→8v: step=2, indices=[0,2,4,6,8,10,12,14]
    # For 8v→4v: step=2, indices=[0,2,4,6]

    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {args.device}")
    print(f"\nConfiguration:")
    print(f"  Teacher views: {args.num_views}")
    print(f"  Student views: {len(student_indices)} (indices: {student_indices})")
    print(f"  Output layers: {args.output_layers}")
    print(f"  LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"  LoRA layers: {args.lora_layers_start}-23")

    # Create dataset
    print("\nCreating dataset...")
    train_dataset = VGGTDistillDataset(
        dataset_name=args.dataset,
        num_views=args.num_views,
        image_size=args.image_size,
        student_indices=student_indices,
        augment=True,
        samples_per_scene=args.samples_per_scene,
        seed=args.seed,
        seeds_list=args.seeds_list,
        first_frame_ref=args.first_frame_ref,
        subset_sampling=args.subset_sampling,
        subset_ratio=args.subset_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Train samples: {len(train_dataset)}")

    # Create models
    print("\nCreating models...")
    lora_layers = list(range(args.lora_layers_start, 24))

    teacher = VGGTTeacherModel(
        model_name=args.model_name,
        output_layers=args.output_layers,
    ).to(args.device)

    student = VGGTStudentModel(
        model_name=args.model_name,
        output_layers=args.output_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_camera_token=args.train_camera_token,
        lora_layers=lora_layers,
    ).to(args.device)

    # Create loss function
    target_layer = args.output_layers[-1]  # Use last output layer for loss
    if args.all_token_softmax_kl_cosine:
        print("\nUsing VGGTAllTokenSoftmaxKLCosineLoss (full 2048-dim softmax KL + cosine)")
        criterion = VGGTAllTokenSoftmaxKLCosineLoss(
            student_frame_indices=student_indices,
            target_layer=target_layer,
            kl_weight=args.all_token_kl_weight,
            cos_weight=args.all_token_cos_weight,
        )
    else:
        print("\nUsing VGGTCombinedSoftmaxKLCosineLoss (frame+global per-token softmax KL + cosine)")
        criterion = VGGTCombinedSoftmaxKLCosineLoss(
            student_frame_indices=student_indices,
            target_layer=target_layer,
            frame_kl_weight=args.frame_kl_weight,
            frame_cos_weight=args.frame_cos_weight,
            global_kl_weight=args.global_kl_weight,
            global_cos_weight=args.global_cos_weight,
        )

    # Create optimizer (only LoRA params)
    optimizer = torch.optim.AdamW(
        student.get_trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(
        optimizer,
        args.lr_scheduler,
        num_training_steps,
        args.warmup_steps,
        len(train_loader),
    )

    # Create scaler for mixed precision
    scaler = GradScaler(enabled=args.use_amp)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, student, optimizer, scheduler, scaler)
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']

    # Create tensorboard writer
    log_dir = os.path.join(args.output_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)

    # Training loop
    print("\nStarting training...")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        global_step = train_epoch(
            teacher, student, train_loader, criterion,
            optimizer, scheduler, scaler,
            epoch, global_step, args, writer,
        )

        # Save epoch checkpoint
        save_checkpoint(
            student, optimizer, scheduler, scaler,
            epoch, global_step, 0.0,
            args.output_dir,
            f"epoch_{epoch}.pt",
        )

    print("\nTraining complete!")
    writer.close()


if __name__ == '__main__':
    main()
