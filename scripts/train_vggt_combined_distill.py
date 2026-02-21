#!/usr/bin/env python3
"""
VGGT Combined Feature + Output Distillation: 8v teacher -> 4v student (LoRA).

Single-pass design:
- Run aggregator once
- Reuse the same aggregator outputs for both feature distillation and output distillation
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src", "vggt"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src", "vggt", "vggt", "training"))

from train_distill_vggt import (  # noqa: E402
    VGGTAllTokenSoftmaxKLCosineLoss,
    VGGTCrossFrameRKDAngleLoss,
)
from train_vggt_output_distill import construct_gt_batch  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.training.loss import MultitaskLoss  # noqa: E402
from vggt.vggt.distillation.dataset import VGGTDistillDataset  # noqa: E402
from vggt.vggt.distillation.models import VGGTDistillationOutput  # noqa: E402


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
    if scheduler_type == "step":
        from torch.optim.lr_scheduler import StepLR

        return StepLR(optimizer, step_size=max(1, steps_per_epoch or num_training_steps), gamma=0.7)
    if scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR

        return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)
    if scheduler_type == "constant":
        from torch.optim.lr_scheduler import ConstantLR

        return ConstantLR(optimizer, factor=1.0, total_iters=num_training_steps)
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def _extract_distill_output(
    output_list: List[torch.Tensor],
    output_layers: List[int],
    embed_dim: int,
) -> VGGTDistillationOutput:
    layer_features: Dict[int, torch.Tensor] = {}
    frame_features: Dict[int, torch.Tensor] = {}
    global_features: Dict[int, torch.Tensor] = {}
    camera_tokens: Dict[int, torch.Tensor] = {}

    for layer_idx in output_layers:
        if layer_idx >= len(output_list):
            raise ValueError(f"Layer {layer_idx} out of range (max {len(output_list) - 1})")

        features = output_list[layer_idx]
        layer_features[layer_idx] = features
        frame_features[layer_idx] = features[..., :embed_dim]
        global_features[layer_idx] = features[..., embed_dim:]
        camera_tokens[layer_idx] = features[:, :, 0, :]

    return VGGTDistillationOutput(
        layer_features=layer_features,
        frame_features=frame_features,
        global_features=global_features,
        camera_tokens=camera_tokens,
    )


def _run_heads(
    output_list: List[torch.Tensor],
    images: torch.Tensor,
    patch_start_idx: int,
    camera_head: Optional[nn.Module],
    depth_head: Optional[nn.Module],
    point_head: Optional[nn.Module],
) -> Dict:
    predictions = {}

    with torch.cuda.amp.autocast(enabled=False):
        if camera_head is not None:
            pose_enc_list = camera_head(output_list)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list

        if depth_head is not None:
            depth, depth_conf = depth_head(output_list, images=images, patch_start_idx=patch_start_idx)
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if point_head is not None:
            world_points, world_points_conf = point_head(output_list, images=images, patch_start_idx=patch_start_idx)
            predictions["world_points"] = world_points
            predictions["world_points_conf"] = world_points_conf

    return predictions


class CombinedTeacher(nn.Module):
    """Frozen VGGT teacher with single-pass dual-output forward."""

    def __init__(
        self,
        model_name: str = "facebook/vggt-1b",
        output_layers: Optional[List[int]] = None,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.output_layers = output_layers or [19, 23]
        self.embed_dim = embed_dim

        print(f"Loading teacher model: {model_name}")
        self.vggt = VGGT.from_pretrained(model_name)

        for param in self.vggt.parameters():
            param.requires_grad = False

        self.eval()
        total = sum(p.numel() for p in self.vggt.parameters())
        print(f"Teacher parameters: {total:,} (all frozen)")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[VGGTDistillationOutput, Dict]:
        output_list, patch_start_idx = self.vggt.aggregator(images)
        distill_output = _extract_distill_output(output_list, self.output_layers, self.embed_dim)
        predictions = _run_heads(
            output_list=output_list,
            images=images,
            patch_start_idx=patch_start_idx,
            camera_head=self.vggt.camera_head,
            depth_head=self.vggt.depth_head,
            point_head=self.vggt.point_head,
        )
        return distill_output, predictions


class CombinedStudent(nn.Module):
    """VGGT + PEFT LoRA on aggregator. Heads frozen. Single-pass dual-output forward."""

    def __init__(
        self,
        model_name: str = "facebook/vggt-1b",
        output_layers: Optional[List[int]] = None,
        embed_dim: int = 1024,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        lora_layers: Optional[List[int]] = None,
        train_camera_token: bool = True,
    ):
        super().__init__()
        self.output_layers = output_layers or [19, 23]
        self.embed_dim = embed_dim
        self.lora_layers = lora_layers or list(range(0, 24))

        print(f"Loading student model: {model_name}")
        self.vggt = VGGT.from_pretrained(model_name)

        for param in self.vggt.parameters():
            param.requires_grad = False

        print(
            f"Applying PEFT LoRA (rank={lora_rank}, alpha={lora_alpha}) "
            f"to layers {self.lora_layers[0]}-{self.lora_layers[-1]}"
        )
        target_modules = []
        for layer_idx in self.lora_layers:
            for block_type in ("frame_blocks", "global_blocks"):
                target_modules.extend(
                    [
                        f"aggregator.{block_type}.{layer_idx}.attn.qkv",
                        f"aggregator.{block_type}.{layer_idx}.attn.proj",
                        f"aggregator.{block_type}.{layer_idx}.mlp.fc1",
                        f"aggregator.{block_type}.{layer_idx}.mlp.fc2",
                    ]
                )

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.vggt = get_peft_model(self.vggt, lora_config)

        if train_camera_token:
            aggregator = self._get_aggregator()
            if hasattr(aggregator, "camera_token"):
                aggregator.camera_token.requires_grad = True
                print("Camera token is trainable")

        total = sum(p.numel() for p in self.vggt.parameters())
        trainable = sum(p.numel() for p in self.vggt.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for n, p in self.vggt.named_parameters() if p.requires_grad and "lora" in n.lower())
        print(f"Student parameters: {total:,} total, {trainable:,} trainable ({lora_params:,} LoRA)")

    def _get_vggt(self):
        if hasattr(self.vggt, "base_model"):
            return self.vggt.base_model.model
        return self.vggt

    def _get_aggregator(self):
        return self._get_vggt().aggregator

    def forward(self, images: torch.Tensor) -> Tuple[VGGTDistillationOutput, Dict]:
        vggt_model = self._get_vggt()
        output_list, patch_start_idx = vggt_model.aggregator(images)
        distill_output = _extract_distill_output(output_list, self.output_layers, self.embed_dim)
        predictions = _run_heads(
            output_list=output_list,
            images=images,
            patch_start_idx=patch_start_idx,
            camera_head=vggt_model.camera_head,
            depth_head=vggt_model.depth_head,
            point_head=vggt_model.point_head,
        )
        return distill_output, predictions

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.vggt.parameters() if p.requires_grad]

    def save_lora_weights(self, path: str) -> None:
        peft_path = path.replace(".pt", "_peft")
        self.vggt.save_pretrained(peft_path)

        state_dict = {}
        aggregator = self._get_aggregator()
        if hasattr(aggregator, "camera_token") and aggregator.camera_token.requires_grad:
            state_dict["camera_token"] = aggregator.camera_token.data.clone()
        if state_dict:
            torch.save(state_dict, path)
        print(f"Saved LoRA weights to {peft_path}")

    def load_lora_weights(self, path: str) -> None:
        peft_path = path.replace(".pt", "_peft")
        if os.path.exists(peft_path) and hasattr(self.vggt, "load_adapter"):
            self.vggt.load_adapter(peft_path, adapter_name="default")
            if hasattr(self.vggt, "set_adapter"):
                self.vggt.set_adapter("default")
            print(f"Loaded LoRA weights from {peft_path}")

        if os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
            if "camera_token" in state_dict:
                aggregator = self._get_aggregator()
                if hasattr(aggregator, "camera_token"):
                    aggregator.camera_token.data.copy_(state_dict["camera_token"])
                    print("Loaded camera token")


def save_checkpoint(student, optimizer, scheduler, scaler, epoch, global_step, loss, output_dir, filename="checkpoint.pt"):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, filename)
    lora_path = os.path.join(output_dir, filename.replace(".pt", "_lora.pt"))

    student.save_lora_weights(lora_path)

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict(),
            "lora_path": lora_path,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")

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


def _to_float(value):
    if torch.is_tensor(value):
        return value.detach().item()
    return float(value)


def train_epoch(
    teacher,
    student,
    train_loader,
    feat_criterion,
    rkd_criterion,
    output_criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    args,
    writer=None,
):
    student.train()
    teacher.eval()

    device = args.device
    grad_accum = args.gradient_accumulation_steps
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    step = args.num_views // args.student_views
    even_indices = [i * step for i in range(args.student_views)]

    for batch_idx, batch in enumerate(train_loader):
        teacher_images = batch["teacher_images"].to(device)
        student_images = batch["student_images"].to(device)

        with autocast(enabled=args.use_amp):
            with torch.no_grad():
                teacher_distill, teacher_preds = teacher(teacher_images)

            student_distill, student_preds = student(student_images)

            feat_loss, feat_details = feat_criterion(teacher_distill, student_distill)
            rkd_loss, rkd_details = rkd_criterion(teacher_distill, student_distill)
            feature_total = feat_loss + args.rkd_weight * rkd_loss

            gt_batch = construct_gt_batch(teacher_preds, student_images, even_indices)
            output_loss_dict = output_criterion(student_preds, gt_batch)
            output_total = output_loss_dict["objective"]

            combined_loss = feature_total + args.output_weight * output_total
            loss = combined_loss / grad_accum

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

        feat_loss_val = _to_float(feat_loss)
        rkd_loss_val = _to_float(rkd_loss)
        feature_total_val = _to_float(feature_total)
        output_total_val = _to_float(output_total)
        combined_loss_val = _to_float(combined_loss)

        total_loss += combined_loss_val
        num_batches += 1

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

            detail_parts = [
                f"feat_loss: {feat_loss_val:.4f}",
                f"rkd_loss: {rkd_loss_val:.4f}",
                f"feature_total: {feature_total_val:.4f}",
                f"output_loss: {output_total_val:.4f}",
                f"combined: {combined_loss_val:.4f}",
            ]
            for key in ("all_softmax_kl", "all_softmax_cos"):
                if key in feat_details:
                    detail_parts.append(f"{key}: {_to_float(feat_details[key]):.4f}")
            for key in ("rkd_angle1_loss", "rkd_angle2_loss", "rkd_angle3_loss"):
                if key in rkd_details:
                    detail_parts.append(f"{key}: {_to_float(rkd_details[key]):.4f}")
            for key in (
                "loss_camera",
                "loss_reg_depth",
                "loss_reg_point",
                "loss_conf_depth",
                "loss_conf_point",
                "loss_grad_depth",
                "loss_grad_point",
            ):
                if key in output_loss_dict:
                    detail_parts.append(f"{key}: {_to_float(output_loss_dict[key]):.4f}")
            print(f"  {' | '.join(detail_parts)}")

            if writer:
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/feat_loss", feat_loss_val, global_step)
                writer.add_scalar("train/rkd_loss", rkd_loss_val, global_step)
                writer.add_scalar("train/feature_total", feature_total_val, global_step)
                writer.add_scalar("train/output_loss", output_total_val, global_step)
                writer.add_scalar("train/combined_loss", combined_loss_val, global_step)
                for key, value in feat_details.items():
                    writer.add_scalar(f"train/{key}", _to_float(value), global_step)
                for key, value in rkd_details.items():
                    writer.add_scalar(f"train/{key}", _to_float(value), global_step)
                for key, value in output_loss_dict.items():
                    writer.add_scalar(f"train/{key}", _to_float(value), global_step)

        if args.save_interval > 0 and global_step > 0 and global_step % args.save_interval == 0:
            save_checkpoint(
                student,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                total_loss / num_batches,
                args.output_dir,
                f"checkpoint_step{global_step}.pt",
            )

    return global_step


def main():
    parser = argparse.ArgumentParser(description="VGGT Combined Feature + Output Distillation")

    # Data
    parser.add_argument("--dataset", type=str, default="eth3d")
    parser.add_argument("--samples_per_scene", type=int, default=5)
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
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--output_layers", type=int, nargs="+", default=[19, 23])
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_layers_start", type=int, default=0)
    parser.add_argument("--train_camera_token", action="store_true", default=True)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=["cosine", "linear", "constant", "step", "none"],
    )
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/vggt_combined_distill")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_amp", action="store_true", default=False)

    # Feature distillation loss
    parser.add_argument("--all_token_kl_weight", type=float, default=1.0)
    parser.add_argument("--all_token_cos_weight", type=float, default=2.0)
    parser.add_argument("--rkd_weight", type=float, default=2.0)
    parser.add_argument("--rkd_topk", type=int, default=4)
    parser.add_argument("--rkd_num_ref_samples", type=int, default=256)
    parser.add_argument("--rkd_num_shared_samples", type=int, default=256)
    parser.add_argument("--rkd_angle1_weight", type=float, default=1.0)
    parser.add_argument("--rkd_angle2_weight", type=float, default=1.0)
    parser.add_argument("--rkd_angle3_weight", type=float, default=1.0)
    parser.add_argument("--rkd_shared_chunk_size", type=int, default=64)

    # Output distillation loss
    parser.add_argument("--output_weight", type=float, default=1.0)
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

    print(f"Using device: {args.device}")
    print("\nConfiguration:")
    print(f"  Teacher views: {args.num_views}")
    print(f"  Student views: {args.student_views} (indices: {student_indices})")
    print(f"  Output layers: {args.output_layers}")
    print(
        f"  Feature loss: all-token KL (w={args.all_token_kl_weight}) + "
        f"cos (w={args.all_token_cos_weight}) + RKD (w={args.rkd_weight})"
    )
    print(
        f"  Output loss: camera (w={args.camera_weight}) + depth (w={args.depth_weight}) + "
        f"point (w={args.point_weight}), output_weight={args.output_weight}"
    )

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

    print("\nCreating models...")
    lora_layers = list(range(args.lora_layers_start, 24))

    teacher = CombinedTeacher(
        model_name=args.model_name,
        output_layers=args.output_layers,
        embed_dim=args.embed_dim,
    ).to(args.device)

    student = CombinedStudent(
        model_name=args.model_name,
        output_layers=args.output_layers,
        embed_dim=args.embed_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=lora_layers,
        train_camera_token=args.train_camera_token,
    ).to(args.device)

    print("\nCreating losses...")
    target_layer = args.output_layers[-1]
    feat_criterion = VGGTAllTokenSoftmaxKLCosineLoss(
        student_frame_indices=student_indices,
        target_layer=target_layer,
        kl_weight=args.all_token_kl_weight,
        cos_weight=args.all_token_cos_weight,
    )
    rkd_criterion = VGGTCrossFrameRKDAngleLoss(
        student_frame_indices=student_indices,
        num_teacher_views=args.num_views,
        target_layer=target_layer,
        topk=args.rkd_topk,
        num_ref_samples=args.rkd_num_ref_samples,
        num_shared_samples=args.rkd_num_shared_samples,
        angle1_weight=args.rkd_angle1_weight,
        angle2_weight=args.rkd_angle2_weight,
        angle3_weight=args.rkd_angle3_weight,
        shared_chunk_size=args.rkd_shared_chunk_size,
    )

    output_criterion = MultitaskLoss(
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

    optimizer = torch.optim.AdamW(student.get_trainable_params(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, args.lr_scheduler, num_training_steps, args.warmup_steps, len(train_loader))
    scaler = GradScaler(enabled=args.use_amp)

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, student, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]

    log_dir = os.path.join(args.output_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir)

    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 60}")

        train_loader.dataset.current_epoch = epoch

        global_step = train_epoch(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            feat_criterion=feat_criterion,
            rkd_criterion=rkd_criterion,
            output_criterion=output_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            args=args,
            writer=writer,
        )

        save_checkpoint(
            student,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            0.0,
            args.output_dir,
            f"epoch_{epoch}.pt",
        )

    print("\nTraining complete!")
    writer.close()


if __name__ == "__main__":
    main()
