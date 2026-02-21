#!/usr/bin/env python3
"""
VGGT Output-Level Distillation: 8v Teacher → 4v Student (LoRA).

For each scene: teacher (frozen VGGT) processes 8 views, extract even-indexed 4v
predictions as "GT". Student (VGGT + LoRA on encoder) processes the same 4 images.
MultitaskLoss (camera + depth + point) between student and teacher's extracted 4v.
Only LoRA weights are updated.

Usage:
    python scripts/train_vggt_output_distill.py --dataset eth3d --epochs 10
    python scripts/train_vggt_output_distill.py --dataset eth3d --debug
    python scripts/train_vggt_output_distill.py --dataset eth3d --resume ./checkpoints/latest.pt
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

# Path setup
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'vggt'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'vggt', 'vggt', 'training'))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.training.loss import MultitaskLoss
from vggt.vggt.distillation.dataset import VGGTDistillDataset
from peft import LoraConfig, get_peft_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer, scheduler_type, num_training_steps, warmup_steps, steps_per_epoch=None):
    if scheduler_type in (None, "none"):
        return None
    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_scheduler = None
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - warmup_steps, eta_min=1e-6)
        if warmup_scheduler is not None:
            return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        return cosine_scheduler
    elif scheduler_type == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=max(1, steps_per_epoch or num_training_steps), gamma=0.7)
    elif scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)
    elif scheduler_type == "constant":
        from torch.optim.lr_scheduler import ConstantLR
        return ConstantLR(optimizer, factor=1.0, total_iters=num_training_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------
class OutputDistillTeacher(nn.Module):
    """Frozen VGGT that runs full forward (aggregator + all heads)."""

    def __init__(self, model_name: str = "facebook/vggt-1b"):
        super().__init__()
        print(f"Loading teacher model: {model_name}")
        self.vggt = VGGT.from_pretrained(model_name)
        for param in self.vggt.parameters():
            param.requires_grad = False
        self.eval()
        total = sum(p.numel() for p in self.vggt.parameters())
        print(f"Teacher parameters: {total:,} (all frozen)")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict:
        return self.vggt(images)


# ---------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------
class OutputDistillStudent(nn.Module):
    """VGGT + PEFT LoRA on aggregator. Heads frozen. Full forward pass."""

    def __init__(
        self,
        model_name: str = "facebook/vggt-1b",
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        lora_layers: Optional[List[int]] = None,
        train_camera_token: bool = True,
    ):
        super().__init__()
        self.lora_layers = lora_layers or list(range(12, 24))

        print(f"Loading student model: {model_name}")
        self.vggt = VGGT.from_pretrained(model_name)

        # Freeze everything
        for param in self.vggt.parameters():
            param.requires_grad = False

        # Apply LoRA to aggregator
        print(f"Applying PEFT LoRA (rank={lora_rank}, alpha={lora_alpha}) to layers {self.lora_layers[0]}-{self.lora_layers[-1]}")
        target_modules = []
        for layer_idx in self.lora_layers:
            for block_type in ("frame_blocks", "global_blocks"):
                target_modules.extend([
                    f"aggregator.{block_type}.{layer_idx}.attn.qkv",
                    f"aggregator.{block_type}.{layer_idx}.attn.proj",
                    f"aggregator.{block_type}.{layer_idx}.mlp.fc1",
                    f"aggregator.{block_type}.{layer_idx}.mlp.fc2",
                ])

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.vggt = get_peft_model(self.vggt, lora_config)

        # Optionally make camera token trainable
        if train_camera_token:
            agg = self._get_aggregator()
            if hasattr(agg, "camera_token"):
                agg.camera_token.requires_grad = True
                print("Camera token is trainable")

        total = sum(p.numel() for p in self.vggt.parameters())
        trainable = sum(p.numel() for p in self.vggt.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for n, p in self.vggt.named_parameters() if p.requires_grad and "lora" in n.lower())
        print(f"Student parameters: {total:,} total, {trainable:,} trainable ({lora_params:,} LoRA)")

    def _get_aggregator(self):
        if hasattr(self.vggt, "base_model"):
            return self.vggt.base_model.model.aggregator
        return self.vggt.aggregator

    def _get_vggt(self):
        if hasattr(self.vggt, "base_model"):
            return self.vggt.base_model.model
        return self.vggt

    def forward(self, images: torch.Tensor) -> Dict:
        return self._get_vggt()(images)

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.vggt.parameters() if p.requires_grad]

    def save_lora_weights(self, path: str) -> None:
        peft_path = path.replace(".pt", "_peft")
        self.vggt.save_pretrained(peft_path)
        state_dict = {}
        agg = self._get_aggregator()
        if hasattr(agg, "camera_token") and agg.camera_token.requires_grad:
            state_dict["camera_token"] = agg.camera_token.data.clone()
        if state_dict:
            torch.save(state_dict, path)
        print(f"Saved LoRA weights to {peft_path}")

    def load_lora_weights(self, path: str) -> None:
        peft_path = path.replace(".pt", "_peft")
        if os.path.exists(peft_path):
            if hasattr(self.vggt, "load_adapter"):
                self.vggt.load_adapter(peft_path, adapter_name="default")
                if hasattr(self.vggt, "set_adapter"):
                    self.vggt.set_adapter("default")
                print(f"Loaded LoRA weights from {peft_path}")
        if os.path.exists(path):
            sd = torch.load(path, map_location="cpu")
            if "camera_token" in sd:
                agg = self._get_aggregator()
                if hasattr(agg, "camera_token"):
                    agg.camera_token.data.copy_(sd["camera_token"])
                    print("Loaded camera token")


# ---------------------------------------------------------------------------
# GT batch construction
# ---------------------------------------------------------------------------
def construct_gt_batch(
    teacher_preds: Dict,
    student_images: torch.Tensor,
    even_indices: List[int] = None,
) -> Dict:
    """Build a GT batch dict from teacher's even-indexed 4v predictions."""
    even_indices = even_indices or [0, 2, 4, 6]
    B, S, C, H, W = student_images.shape

    # Extract even-indexed predictions
    teacher_pose_enc = teacher_preds["pose_enc"][:, even_indices]          # [B, 4, 9]
    teacher_depth = teacher_preds["depth"][:, even_indices]                # [B, 4, H', W', 1]
    teacher_points = teacher_preds["world_points"][:, even_indices]        # [B, 4, H', W', 3]

    # Decode pose_enc → extrinsics / intrinsics
    teacher_ext, teacher_int = pose_encoding_to_extri_intri(
        teacher_pose_enc, image_size_hw=(H, W)
    )

    # All-True point masks (teacher predictions exist everywhere)
    depth_H, depth_W = teacher_depth.shape[2], teacher_depth.shape[3]
    point_masks = torch.ones(B, len(even_indices), depth_H, depth_W,
                             dtype=torch.bool, device=student_images.device)

    # Also extract pose_enc_list for each refinement stage
    teacher_pose_enc_list = []
    for stage_enc in teacher_preds["pose_enc_list"]:
        teacher_pose_enc_list.append(stage_enc[:, even_indices])

    return {
        "images": student_images,
        "extrinsics": teacher_ext.detach(),                     # [B, 4, 3, 4]
        "intrinsics": teacher_int.detach(),                     # [B, 4, 3, 3]
        "depths": teacher_depth.squeeze(-1).detach(),           # [B, 4, H', W']
        "world_points": teacher_points.detach(),                # [B, 4, H', W', 3]
        "point_masks": point_masks,                             # [B, 4, H', W']
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(student, optimizer, scheduler, scaler, epoch, global_step, loss, output_dir, filename="checkpoint.pt"):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, filename)
    lora_path = os.path.join(output_dir, filename.replace(".pt", "_lora.pt"))

    student.save_lora_weights(lora_path)

    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "lora_path": lora_path,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Also save as latest
    latest_path = os.path.join(output_dir, "latest.pt")
    latest_lora = os.path.join(output_dir, "latest_lora.pt")
    torch.save(torch.load(ckpt_path, map_location="cpu"), latest_path)
    student.save_lora_weights(latest_lora)
    return ckpt_path


def load_checkpoint(path, student, optimizer, scheduler, scaler):
    print(f"Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location="cpu")
    if "lora_path" in ckpt and os.path.exists(ckpt["lora_path"]):
        student.load_lora_weights(ckpt["lora_path"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(
    teacher, student, train_loader, loss_fn, optimizer, scheduler, scaler,
    epoch, global_step, args, writer=None,
):
    student.train()
    teacher.eval()

    device = args.device
    grad_accum = args.gradient_accumulation_steps
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    # Compute student frame indices
    step = args.num_views // args.student_views
    even_indices = [i * step for i in range(args.student_views)]

    for batch_idx, batch in enumerate(train_loader):
        teacher_images = batch["teacher_images"].to(device)   # [B, 8, 3, H, W]
        student_images = batch["student_images"].to(device)   # [B, 4, 3, H, W]

        with autocast(enabled=args.use_amp):
            # Teacher forward (no grad, full VGGT)
            with torch.no_grad():
                teacher_preds = teacher(teacher_images)

            # Construct GT batch from teacher's even-indexed 4v
            gt_batch = construct_gt_batch(teacher_preds, student_images, even_indices)

            # Student forward (grad through LoRA)
            student_preds = student(student_images)

            # MultitaskLoss
            loss_dict = loss_fn(student_preds, gt_batch)
            loss = loss_dict["objective"]
            loss = loss / grad_accum

        # Backward
        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.get_trainable_params(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

        total_loss += loss.item() * grad_accum
        num_batches += 1

        # Logging
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            speed = (batch_idx + 1) * args.batch_size / elapsed

            print(
                f"Epoch {epoch} | Step {global_step} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                f"Speed: {speed:.1f} samples/s"
            )
            detail_parts = []
            for k in ("loss_camera", "loss_T", "loss_R", "loss_FL",
                       "loss_conf_depth", "loss_reg_depth", "loss_grad_depth",
                       "loss_conf_point", "loss_reg_point", "loss_grad_point"):
                if k in loss_dict:
                    v = loss_dict[k]
                    v = v.item() if torch.is_tensor(v) else v
                    detail_parts.append(f"{k}: {v:.4f}")
            if detail_parts:
                print(f"  {' | '.join(detail_parts)}")

            if writer:
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                for k, v in loss_dict.items():
                    v = v.item() if torch.is_tensor(v) else v
                    writer.add_scalar(f"train/{k}", v, global_step)

        # Periodic save
        if args.save_interval > 0 and global_step > 0 and global_step % args.save_interval == 0:
            save_checkpoint(student, optimizer, scheduler, scaler, epoch, global_step,
                            total_loss / num_batches, args.output_dir, f"checkpoint_step{global_step}.pt")

    return global_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VGGT Output-Level Distillation")

    # Data
    parser.add_argument("--dataset", type=str, default="eth3d")
    parser.add_argument("--samples_per_scene", type=int, default=4)
    parser.add_argument("--seeds_list", type=int, nargs="+", default=None)
    parser.add_argument("--first_frame_ref", action="store_true")
    parser.add_argument("--subset_sampling", action="store_true")
    parser.add_argument("--subset_ratio", type=float, default=0.05)
    parser.add_argument("--stride_sampling", action="store_true")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--paired_sampling", action="store_true")
    parser.add_argument("--paired_gap", type=int, default=3)
    parser.add_argument("--fixed_subset_seed", type=int, default=None)
    parser.add_argument("--fixed_subset_max_frames", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--student_views", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=504)

    # Model
    parser.add_argument("--model_name", type=str, default="facebook/vggt-1b")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_layers_start", type=int, default=0)
    parser.add_argument("--train_camera_token", action="store_true", default=True)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["cosine", "linear", "constant", "step", "none"])
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/vggt_output_distill")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_amp", action="store_true", default=True)

    # Loss weights
    parser.add_argument("--camera_weight", type=float, default=5.0)
    parser.add_argument("--camera_loss_type", type=str, default="l1")
    parser.add_argument("--camera_gamma", type=float, default=0.6)
    parser.add_argument("--weight_trans", type=float, default=1.0)
    parser.add_argument("--weight_rot", type=float, default=1.0)
    parser.add_argument("--weight_focal", type=float, default=0.5)
    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--depth_gradient_loss", type=str, default="grad")
    parser.add_argument("--depth_valid_range", type=float, default=0.98)
    parser.add_argument("--point_weight", type=float, default=1.0)
    parser.add_argument("--point_gradient_loss", type=str, default="normal")
    parser.add_argument("--point_valid_range", type=float, default=0.98)

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.debug:
        args.epochs = 1
        args.batch_size = 1
        args.samples_per_scene = 1
        args.log_interval = 1
        args.save_interval = 0

    if args.student_views is None:
        args.student_views = args.num_views // 2

    step = args.num_views // args.student_views
    student_indices = [i * step for i in range(args.student_views)]

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Teacher views: {args.num_views}, Student views: {args.student_views} (indices: {student_indices})")

    # Dataset
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
        stride_sampling=args.stride_sampling,
        stride=args.stride,
        paired_sampling=args.paired_sampling,
        paired_gap=args.paired_gap,
        fixed_subset_seed=args.fixed_subset_seed,
        fixed_subset_max_frames=args.fixed_subset_max_frames,
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

    # Models
    print("\nCreating models...")
    teacher = OutputDistillTeacher(model_name=args.model_name).to(args.device)
    lora_layers = list(range(args.lora_layers_start, 24))
    student = OutputDistillStudent(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=lora_layers,
        train_camera_token=args.train_camera_token,
    ).to(args.device)

    # Loss
    print("\nCreating MultitaskLoss...")
    loss_fn = MultitaskLoss(
        camera={
            "weight": args.camera_weight,
            "loss_type": args.camera_loss_type,
            "gamma": args.camera_gamma,
            "weight_trans": args.weight_trans,
            "weight_rot": args.weight_rot,
            "weight_focal": args.weight_focal,
        },
        depth={
            "weight": args.depth_weight,
            "gradient_loss_fn": args.depth_gradient_loss,
            "valid_range": args.depth_valid_range,
        },
        point={
            "weight": args.point_weight,
            "gradient_loss_fn": args.point_gradient_loss,
            "valid_range": args.point_valid_range,
        },
    )

    # Optimizer
    optimizer = torch.optim.AdamW(student.get_trainable_params(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, args.lr_scheduler, num_training_steps, args.warmup_steps, len(train_loader))
    scaler = GradScaler(enabled=args.use_amp)

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, student, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]

    # Tensorboard
    log_dir = os.path.join(args.output_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir)

    # Train
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        train_loader.dataset.current_epoch = epoch

        global_step = train_epoch(
            teacher, student, train_loader, loss_fn,
            optimizer, scheduler, scaler,
            epoch, global_step, args, writer,
        )

        save_checkpoint(student, optimizer, scheduler, scaler, epoch, global_step, 0.0,
                        args.output_dir, f"epoch_{epoch}.pt")

    print("\nTraining complete!")
    writer.close()


if __name__ == "__main__":
    main()
