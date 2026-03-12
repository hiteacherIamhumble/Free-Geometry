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
    eta_min: float = 1e-6,
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
            eta_min=eta_min,
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


class VGGTPatchHuberCosineLoss(nn.Module):
    """
    Per-patch Huber + cosine loss on full VGGT tokens (DA3-style).
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        target_layers: Optional[List[int]] = None,
        huber_weight: float = 1.0,
        cos_weight: float = 2.0,
        delta: float = 1.0,
    ):
        super().__init__()
        self.target_layers = target_layers or [23]
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.huber_weight = huber_weight
        self.cos_weight = cos_weight
        self.huber = nn.SmoothL1Loss(reduction='mean', beta=delta)
        print(
            f"VGGTPatchHuberCosineLoss: layers={self.target_layers}, "
            f"huber_w={huber_weight}, cos_w={cos_weight}, delta={delta}"
        )

    def forward(
        self,
        teacher_output: VGGTDistillationOutput,
        student_output: VGGTDistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_huber = 0.0
        total_cos = 0.0

        for layer in self.target_layers:
            teacher_all = teacher_output.layer_features[layer]  # [B,8,P,D]
            student_all = student_output.layer_features[layer]  # [B,4,P,D]
            teacher_selected = teacher_all[:, self.student_frame_indices, :, :].detach()

            huber = self.huber(student_all, teacher_selected)
            teacher_norm = torch.nn.functional.normalize(teacher_selected, dim=-1)
            student_norm = torch.nn.functional.normalize(student_all, dim=-1)
            cos = (teacher_norm * student_norm).sum(dim=-1).mean()

            total_huber = total_huber + huber
            total_cos = total_cos + cos

        n = len(self.target_layers)
        avg_huber = total_huber / n
        avg_cos = total_cos / n

        loss = self.huber_weight * avg_huber + self.cos_weight * (1.0 - avg_cos)
        return loss, {
            'patch_huber': avg_huber.item(),
            'patch_huber_cos': avg_cos.item(),
            'patch_huber_total': loss.item(),
            'total_loss': loss.item(),
        }


class VGGTCrossFrameRKDAngleLoss(nn.Module):
    """
    Cross-Frame RKD Angle-Wise Loss for VGGT distillation.

    Based on RKD (Park et al., CVPR 2019). Transfers teacher's knowledge about
    extra frames (those the student does NOT see) via angle-wise relational
    knowledge distillation.

    For each sampled reference patch p in Frame 0 and shared patch q in a shared
    frame, we find the top-K most similar patches to ref_t[p] across the teacher's
    extra frames. Three angles are computed from the triplet (ref, shared, sim_high)
    and the student is trained to match the teacher's angle structure.
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        num_teacher_views: int = 8,
        target_layer: int = 23,
        topk: int = 4,
        num_ref_samples: int = 128,
        num_shared_samples: int = 128,
        angle1_weight: float = 1.0,
        angle2_weight: float = 1.0,
        angle3_weight: float = 1.0,
        huber_delta: float = 1.0,
        shared_chunk_size: int = 64,
        selection_mode: str = 'topk',
    ):
        super().__init__()
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.num_teacher_views = num_teacher_views
        self.target_layer = target_layer
        self.topk = topk
        self.num_ref_samples = num_ref_samples
        self.num_shared_samples = num_shared_samples
        self.angle1_weight = angle1_weight
        self.angle2_weight = angle2_weight
        self.angle3_weight = angle3_weight
        self.huber_loss = nn.HuberLoss(reduction='none', delta=huber_delta)
        self.shared_chunk_size = shared_chunk_size
        self.selection_mode = selection_mode

        # Extra frame indices (teacher-only frames)
        all_teacher_indices = set(range(num_teacher_views))
        student_set = set(self.student_frame_indices)
        self.extra_frame_indices = sorted(all_teacher_indices - student_set)

        # Reference frame = first student frame; shared = remaining student frames
        self.ref_frame_teacher_idx = self.student_frame_indices[0]
        self.shared_frame_teacher_indices = self.student_frame_indices[1:]

        # Map teacher frame index -> student frame index
        self.teacher_to_student = {
            t_idx: s_idx for s_idx, t_idx in enumerate(self.student_frame_indices)
        }

        print(f"VGGTCrossFrameRKDAngleLoss: layer={target_layer}, topk={topk}, selection={selection_mode}")
        print(f"  ref_samples={num_ref_samples}, shared_samples={num_shared_samples}, shared_chunk={shared_chunk_size}")
        print(f"  angle weights: a1={angle1_weight}, a2={angle2_weight}, a3={angle3_weight}")
        print(f"  extra frames (teacher-only): {self.extra_frame_indices}")
        print(f"  shared frames: {self.shared_frame_teacher_indices}")

    @staticmethod
    def _cos_angle(vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine of angle between two vectors (already relative to vertex).

        Args:
            vec_a: [..., D]
            vec_b: [..., D]
        Returns:
            [...] cosine similarity values in [-1, 1]
        """
        a_norm = torch.nn.functional.normalize(vec_a, dim=-1, eps=1e-8)
        b_norm = torch.nn.functional.normalize(vec_b, dim=-1, eps=1e-8)
        return (a_norm * b_norm).sum(dim=-1)

    def forward(
        self,
        teacher_output: VGGTDistillationOutput,
        student_output: VGGTDistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_feats = teacher_output.layer_features[self.target_layer]  # [B, S_t, P, D]
        student_feats = student_output.layer_features[self.target_layer]  # [B, S_s, P, D]

        B, S_t, P, D = teacher_feats.shape

        # --- 1. Subsample positions upfront ---
        num_ref = min(self.num_ref_samples, P)
        num_shared = min(self.num_shared_samples, P)
        ref_perm = torch.randperm(P, device=teacher_feats.device)[:num_ref]
        shared_perm = torch.randperm(P, device=teacher_feats.device)[:num_shared]

        # --- 2. Extract only the slices we need (avoid holding full [B,S,P,D]) ---
        ref_t_sampled = teacher_feats[:, self.ref_frame_teacher_idx, ref_perm, :].detach()  # [B, num_ref, D]
        ref_s_idx = self.teacher_to_student[self.ref_frame_teacher_idx]
        ref_s_sampled = student_feats[:, ref_s_idx, ref_perm, :]  # [B, num_ref, D]

        # --- 3. Select patches across extra frames (no grad) ---
        with torch.no_grad():
            # Build extra features: [B, E*P, D] — only needed temporarily
            extra_t_list = [teacher_feats[:, eidx, :, :] for eidx in self.extra_frame_indices]
            extra_t = torch.cat(extra_t_list, dim=1)  # [B, E*P, D]
            del extra_t_list

            if self.selection_mode == 'random':
                EP = extra_t.shape[1]
                rand_indices = torch.randint(0, EP, (B, num_ref, self.topk), device=teacher_feats.device)
                sim_high_t = torch.zeros(B, num_ref, self.topk, D,
                                         device=teacher_feats.device, dtype=teacher_feats.dtype)
                for b in range(B):
                    flat_idx = rand_indices[b].reshape(-1)
                    sim_high_t[b] = extra_t[b, flat_idx, :].reshape(num_ref, self.topk, D)
                del extra_t, rand_indices

            elif self.selection_mode == 'mixed':
                ref_t_norm = torch.nn.functional.normalize(ref_t_sampled, dim=-1)
                extra_t_norm = torch.nn.functional.normalize(extra_t, dim=-1)
                sim_matrix = torch.bmm(ref_t_norm, extra_t_norm.transpose(1, 2))
                del ref_t_norm, extra_t_norm

                k_top = self.topk // 2
                k_bot = self.topk - k_top
                _, topk_indices = sim_matrix.topk(k_top, dim=-1, largest=True)
                _, botk_indices = sim_matrix.topk(k_bot, dim=-1, largest=False)
                combined_indices = torch.cat([topk_indices, botk_indices], dim=-1)
                del sim_matrix, topk_indices, botk_indices

                sim_high_t = torch.zeros(B, num_ref, self.topk, D,
                                         device=teacher_feats.device, dtype=teacher_feats.dtype)
                for b in range(B):
                    flat_idx = combined_indices[b].reshape(-1)
                    sim_high_t[b] = extra_t[b, flat_idx, :].reshape(num_ref, self.topk, D)
                del extra_t, combined_indices

            else:
                ref_t_norm = torch.nn.functional.normalize(ref_t_sampled, dim=-1)
                extra_t_norm = torch.nn.functional.normalize(extra_t, dim=-1)
                sim_matrix = torch.bmm(ref_t_norm, extra_t_norm.transpose(1, 2))
                del ref_t_norm, extra_t_norm

                _, topk_indices = sim_matrix.topk(self.topk, dim=-1)
                del sim_matrix

                sim_high_t = torch.zeros(B, num_ref, self.topk, D,
                                         device=teacher_feats.device, dtype=teacher_feats.dtype)
                for b in range(B):
                    flat_idx = topk_indices[b].reshape(-1)
                    sim_high_t[b] = extra_t[b, flat_idx, :].reshape(num_ref, self.topk, D)
                del extra_t, topk_indices

        sim_high_t = sim_high_t.detach()

        # --- 4. Compute angles (chunked over both ref and shared dims) ---
        # Peak memory per chunk: [B, rc, sc, K, D] × ~4 intermediates in _cos_angle
        sum_angle1 = torch.tensor(0.0, device=teacher_feats.device)
        sum_angle2 = torch.tensor(0.0, device=teacher_feats.device)
        sum_angle3 = torch.tensor(0.0, device=teacher_feats.device)
        total_elements = 0
        cs = self.shared_chunk_size

        for shared_teacher_idx in self.shared_frame_teacher_indices:
            shared_student_idx = self.teacher_to_student[shared_teacher_idx]

            shared_t_full = teacher_feats[:, shared_teacher_idx, shared_perm, :].detach()
            shared_s_full = student_feats[:, shared_student_idx, shared_perm, :]

            for r0 in range(0, num_ref, cs):
                r1 = min(r0 + cs, num_ref)
                rc = r1 - r0

                ref_t_4d = ref_t_sampled[:, r0:r1, :].detach().unsqueeze(2).unsqueeze(3)  # [B,rc,1,1,D]
                ref_s_4d = ref_s_sampled[:, r0:r1, :].unsqueeze(2).unsqueeze(3)
                sim_high_4d = sim_high_t[:, r0:r1, :, :].unsqueeze(2)  # [B,rc,1,K,D]

                for s0 in range(0, num_shared, cs):
                    s1 = min(s0 + cs, num_shared)
                    sc = s1 - s0

                    shared_t_4d = shared_t_full[:, s0:s1, :].unsqueeze(1).unsqueeze(3)  # [B,1,sc,1,D]
                    shared_s_4d = shared_s_full[:, s0:s1, :].unsqueeze(1).unsqueeze(3)

                    n_elem = B * rc * sc * self.topk

                    # Angle 1: vertex = ref
                    a1_t = self._cos_angle(shared_t_4d - ref_t_4d, sim_high_4d - ref_t_4d)
                    a1_s = self._cos_angle(shared_s_4d - ref_s_4d, sim_high_4d - ref_s_4d)
                    sum_angle1 = sum_angle1 + self.huber_loss(a1_s, a1_t.detach()).sum()
                    del a1_t, a1_s

                    # Angle 2: vertex = sim_high
                    a2_t = self._cos_angle(ref_t_4d - sim_high_4d, shared_t_4d - sim_high_4d)
                    a2_s = self._cos_angle(ref_s_4d - sim_high_4d, shared_s_4d - sim_high_4d)
                    sum_angle2 = sum_angle2 + self.huber_loss(a2_s, a2_t.detach()).sum()
                    del a2_t, a2_s

                    # Angle 3: vertex = shared
                    a3_t = self._cos_angle(ref_t_4d - shared_t_4d, sim_high_4d - shared_t_4d)
                    a3_s = self._cos_angle(ref_s_4d - shared_s_4d, sim_high_4d - shared_s_4d)
                    sum_angle3 = sum_angle3 + self.huber_loss(a3_s, a3_t.detach()).sum()
                    del a3_t, a3_s

                    total_elements += n_elem

        # Mean over all elements
        total_angle1_loss = sum_angle1 / total_elements
        total_angle2_loss = sum_angle2 / total_elements
        total_angle3_loss = sum_angle3 / total_elements

        # Weighted combination
        loss = (
            self.angle1_weight * total_angle1_loss
            + self.angle2_weight * total_angle2_loss
            + self.angle3_weight * total_angle3_loss
        )

        return loss, {
            'rkd_angle1_loss': total_angle1_loss.item(),
            'rkd_angle2_loss': total_angle2_loss.item(),
            'rkd_angle3_loss': total_angle3_loss.item(),
            'rkd_total_loss': loss.item(),
        }


class VGGTCrossFrameRKDDistanceLoss(nn.Module):
    """
    Cross-Frame RKD distance-wise loss for VGGT distillation.

    This mirrors DA3 distance RKD: for triplets (ref, shared, sim_high), match
    the per-channel difference profiles for the 3 edges:
      d1 = ref - shared
      d2 = ref - sim_high
      d3 = shared - sim_high
    """

    def __init__(
        self,
        student_frame_indices: Optional[List[int]] = None,
        num_teacher_views: int = 8,
        target_layer: int = 23,
        topk: int = 4,
        num_ref_samples: int = 256,
        num_shared_samples: int = 256,
        d1_weight: float = 1.0,
        d2_weight: float = 1.0,
        d3_weight: float = 1.0,
        shared_chunk_size: int = 64,
        distance_chunk_size: int = 16,
        distance_type: str = "l2",
        normalize_distance: bool = True,
        temperature: float = 1.0,
        distance_mode: str = "kl",
        huber_beta: float = 0.5,
        selection_mode: str = 'topk',
    ):
        super().__init__()
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.num_teacher_views = num_teacher_views
        self.target_layer = target_layer
        self.topk = topk
        self.num_ref_samples = num_ref_samples
        self.num_shared_samples = num_shared_samples
        self.d1_weight = d1_weight
        self.d2_weight = d2_weight
        self.d3_weight = d3_weight
        self.shared_chunk_size = shared_chunk_size
        self.distance_chunk_size = distance_chunk_size
        self.distance_type = distance_type
        self.normalize_distance = normalize_distance
        self.temperature = temperature
        self.distance_mode = distance_mode
        self.huber_beta = huber_beta
        self.selection_mode = selection_mode

        all_teacher_indices = set(range(num_teacher_views))
        student_set = set(self.student_frame_indices)
        self.extra_frame_indices = sorted(all_teacher_indices - student_set)

        self.ref_frame_teacher_idx = self.student_frame_indices[0]
        self.shared_frame_teacher_indices = self.student_frame_indices[1:]

        self.teacher_to_student = {
            t_idx: s_idx for s_idx, t_idx in enumerate(self.student_frame_indices)
        }

        print(
            f"VGGTCrossFrameRKDDistanceLoss: layer={target_layer}, topk={topk}, "
            f"temp={temperature}, mode={distance_mode}, selection={selection_mode}"
        )
        print(
            f"  ref_samples={num_ref_samples}, shared_samples={num_shared_samples}, "
            f"shared_chunk={shared_chunk_size}, distance_chunk={distance_chunk_size}"
        )
        print(f"  distance weights: d1={d1_weight}, d2={d2_weight}, d3={d3_weight}")
        print(f"  distance_type={distance_type}, normalize={normalize_distance}")
        if distance_mode == "huber":
            print(f"  huber_beta={huber_beta}")
        print(f"  extra frames (teacher-only): {self.extra_frame_indices}")
        print(f"  shared frames: {self.shared_frame_teacher_indices}")

    def forward(
        self,
        teacher_output: VGGTDistillationOutput,
        student_output: VGGTDistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_feats = teacher_output.layer_features[self.target_layer]  # [B, S_t, P, D]
        student_feats = student_output.layer_features[self.target_layer]  # [B, S_s, P, D]

        B, _, P, D = teacher_feats.shape

        # 1) Position subsampling.
        num_ref = min(self.num_ref_samples, P)
        num_shared = min(self.num_shared_samples, P)
        ref_perm = torch.randperm(P, device=teacher_feats.device)[:num_ref]
        shared_perm = torch.randperm(P, device=teacher_feats.device)[:num_shared]

        # 2) Gather ref slices.
        ref_t_sampled = teacher_feats[:, self.ref_frame_teacher_idx, ref_perm, :].detach()
        ref_s_idx = self.teacher_to_student[self.ref_frame_teacher_idx]
        ref_s_sampled = student_feats[:, ref_s_idx, ref_perm, :]

        # 3) Select patches over teacher-only frames.
        with torch.no_grad():
            extra_t_list = [teacher_feats[:, eidx, :, :] for eidx in self.extra_frame_indices]
            extra_t = torch.cat(extra_t_list, dim=1)  # [B, E*P, D]

            if self.selection_mode == 'random':
                EP = extra_t.shape[1]
                rand_indices = torch.randint(0, EP, (B, num_ref, self.topk), device=teacher_feats.device)
                sim_high_t = torch.zeros(B, num_ref, self.topk, D,
                                         device=teacher_feats.device, dtype=teacher_feats.dtype)
                for b in range(B):
                    flat_idx = rand_indices[b].reshape(-1)
                    sim_high_t[b] = extra_t[b, flat_idx, :].reshape(num_ref, self.topk, D)
                del extra_t, rand_indices

            elif self.selection_mode == 'mixed':
                ref_t_norm = torch.nn.functional.normalize(ref_t_sampled, dim=-1)
                extra_t_norm = torch.nn.functional.normalize(extra_t, dim=-1)
                sim_matrix = torch.bmm(ref_t_norm, extra_t_norm.transpose(1, 2))
                del ref_t_norm, extra_t_norm

                k_top = self.topk // 2
                k_bot = self.topk - k_top
                _, topk_indices = sim_matrix.topk(k_top, dim=-1, largest=True)
                _, botk_indices = sim_matrix.topk(k_bot, dim=-1, largest=False)
                combined_indices = torch.cat([topk_indices, botk_indices], dim=-1)
                del sim_matrix, topk_indices, botk_indices

                sim_high_t = torch.zeros(B, num_ref, self.topk, D,
                                         device=teacher_feats.device, dtype=teacher_feats.dtype)
                for b in range(B):
                    flat_idx = combined_indices[b].reshape(-1)
                    sim_high_t[b] = extra_t[b, flat_idx, :].reshape(num_ref, self.topk, D)
                del extra_t, combined_indices

            else:
                ref_t_norm = torch.nn.functional.normalize(ref_t_sampled, dim=-1)
                extra_t_norm = torch.nn.functional.normalize(extra_t, dim=-1)
                sim_matrix = torch.bmm(ref_t_norm, extra_t_norm.transpose(1, 2))
                _, topk_indices = sim_matrix.topk(self.topk, dim=-1)

                sim_high_t = torch.zeros(B, num_ref, self.topk, D,
                                         device=teacher_feats.device, dtype=teacher_feats.dtype)
                for b in range(B):
                    flat_idx = topk_indices[b].reshape(-1)
                    sim_high_t[b] = extra_t[b, flat_idx, :].reshape(num_ref, self.topk, D)
                del extra_t, topk_indices

        sim_high_t = sim_high_t.detach()

        # 4) Chunked distance-profile matching.
        N = min(num_ref, num_shared)
        cs = self.distance_chunk_size
        temperature = self.temperature
        huber_outer = nn.SmoothL1Loss(reduction="none", beta=0.5)
        huber_direct = nn.SmoothL1Loss(reduction="none", beta=self.huber_beta)

        sum_d1 = torch.tensor(0.0, device=teacher_feats.device)
        sum_d2 = torch.tensor(0.0, device=teacher_feats.device)
        sum_d3 = torch.tensor(0.0, device=teacher_feats.device)
        total_d1_elements = 0
        total_d2_elements = 0
        total_d3_elements = 0

        ref_t_paired = ref_t_sampled[:, :N, :]
        ref_s_paired = ref_s_sampled[:, :N, :]
        sim_high_paired = sim_high_t[:, :N, :, :]

        for shared_teacher_idx in self.shared_frame_teacher_indices:
            shared_student_idx = self.teacher_to_student[shared_teacher_idx]
            shared_t = teacher_feats[:, shared_teacher_idx, shared_perm[:N], :].detach()
            shared_s = student_feats[:, shared_student_idx, shared_perm[:N], :]

            for c0 in range(0, N, cs):
                c1 = min(c0 + cs, N)
                rt = ref_t_paired[:, c0:c1, :].detach()
                rs = ref_s_paired[:, c0:c1, :]
                st = shared_t[:, c0:c1, :]
                ss = shared_s[:, c0:c1, :]
                sh = sim_high_paired[:, c0:c1, :, :]

                # d1: ref - shared
                diff_d1_t = rt - st
                diff_d1_s = rs - ss

                if self.distance_mode == "huber":
                    loss_d1_chunk = huber_direct(diff_d1_s, diff_d1_t).mean(dim=-1)
                    sum_d1 = sum_d1 + loss_d1_chunk.sum()
                    total_d1_elements += loss_d1_chunk.numel()
                else:
                    log_p = torch.log_softmax(diff_d1_t / temperature, dim=-1)
                    log_q = torch.log_softmax(diff_d1_s / temperature, dim=-1)
                    p = log_p.exp()
                    kl_d1 = (p * (log_p - log_q)).sum(dim=-1)
                    sum_d1 = sum_d1 + huber_outer(kl_d1, torch.zeros_like(kl_d1)).sum()
                    total_d1_elements += kl_d1.numel()

                # d2: ref - sim_high
                diff_d2_t = rt.unsqueeze(2) - sh
                diff_d2_s = rs.unsqueeze(2) - sh

                if self.distance_mode == "huber":
                    loss_d2_chunk = huber_direct(diff_d2_s, diff_d2_t).mean(dim=-1)
                    sum_d2 = sum_d2 + loss_d2_chunk.sum()
                    total_d2_elements += loss_d2_chunk.numel()
                else:
                    log_p = torch.log_softmax(diff_d2_t / temperature, dim=-1)
                    log_q = torch.log_softmax(diff_d2_s / temperature, dim=-1)
                    p = log_p.exp()
                    kl_d2 = (p * (log_p - log_q)).sum(dim=-1)
                    sum_d2 = sum_d2 + huber_outer(kl_d2, torch.zeros_like(kl_d2)).sum()
                    total_d2_elements += kl_d2.numel()

                # d3: shared - sim_high
                diff_d3_t = st.unsqueeze(2) - sh
                diff_d3_s = ss.unsqueeze(2) - sh

                if self.distance_mode == "huber":
                    loss_d3_chunk = huber_direct(diff_d3_s, diff_d3_t).mean(dim=-1)
                    sum_d3 = sum_d3 + loss_d3_chunk.sum()
                    total_d3_elements += loss_d3_chunk.numel()
                else:
                    log_p = torch.log_softmax(diff_d3_t / temperature, dim=-1)
                    log_q = torch.log_softmax(diff_d3_s / temperature, dim=-1)
                    p = log_p.exp()
                    kl_d3 = (p * (log_p - log_q)).sum(dim=-1)
                    sum_d3 = sum_d3 + huber_outer(kl_d3, torch.zeros_like(kl_d3)).sum()
                    total_d3_elements += kl_d3.numel()

        loss_d1 = sum_d1 / max(total_d1_elements, 1)
        loss_d2 = sum_d2 / max(total_d2_elements, 1)
        loss_d3 = sum_d3 / max(total_d3_elements, 1)

        loss = self.d1_weight * loss_d1 + self.d2_weight * loss_d2 + self.d3_weight * loss_d3
        return loss, {
            "rkd_dist_d1_loss": loss_d1.item(),
            "rkd_dist_d2_loss": loss_d2.item(),
            "rkd_dist_d3_loss": loss_d3.item(),
            "rkd_dist_total_loss": loss.item(),
        }


class VGGTOutputDistillLoss(nn.Module):
    """Output-level distillation loss for VGGT: pose_enc L1 + depth L1."""

    def __init__(
        self,
        student_frame_indices: List[int],
        camera_weight: float = 5.0,
        depth_weight: float = 1.0,
    ):
        super().__init__()
        self.student_frame_indices = student_frame_indices
        self.camera_weight = camera_weight
        self.depth_weight = depth_weight

    def forward(
        self,
        teacher_preds: Dict[str, torch.Tensor],
        student_preds: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = next(iter(student_preds.values())).device
        total_loss = torch.tensor(0.0, device=device)
        details: Dict[str, float] = {}

        # Camera loss: L1 on pose_enc [B, S, 9]
        if "pose_enc" in teacher_preds and "pose_enc" in student_preds:
            t_pose = teacher_preds["pose_enc"][:, self.student_frame_indices].detach()
            s_pose = student_preds["pose_enc"]
            cam_loss = (t_pose - s_pose).abs().clamp(max=100).mean()
            total_loss = total_loss + self.camera_weight * cam_loss
            details["output_cam_loss"] = cam_loss.item()

        # Depth loss: L1 on depth [B, S, H, W, 1]
        if "depth" in teacher_preds and "depth" in student_preds:
            t_depth = teacher_preds["depth"][:, self.student_frame_indices].detach()
            s_depth = student_preds["depth"]
            depth_loss = (t_depth - s_depth).abs().clamp(max=100).mean()
            total_loss = total_loss + self.depth_weight * depth_loss
            details["output_depth_loss"] = depth_loss.item()

        details["output_total_loss"] = total_loss.item()
        return total_loss, details


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
    rkd_criterion: Optional[nn.Module] = None,
    rkd_distance_criterion: Optional[nn.Module] = None,
    output_criterion: Optional[nn.Module] = None,
) -> int:
    """Train for one epoch."""
    student.train()
    teacher.eval()

    device = args.device
    log_interval = args.log_interval
    save_interval = args.save_interval
    grad_accum_steps = args.gradient_accumulation_steps
    use_output_loss = output_criterion is not None and args.output_weight > 0.0

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
                if use_output_loss:
                    teacher_output, teacher_preds = teacher.forward_with_preds(teacher_images)
                else:
                    teacher_output = teacher(teacher_images)

            # Student forward
            if use_output_loss:
                student_output, student_preds = student.forward_with_preds(student_images)
            else:
                student_output = student(student_images)

            # Compute feature loss
            loss, loss_details = criterion(teacher_output, student_output)

            # Add cross-frame RKD loss if enabled
            if rkd_criterion is not None:
                rkd_loss, rkd_details = rkd_criterion(teacher_output, student_output)
                loss_details['base_loss'] = loss.item()
                loss_details['rkd_weighted'] = (args.rkd_weight * rkd_loss).item()
                loss = loss + args.rkd_weight * rkd_loss
                loss_details.update(rkd_details)
                loss_details['total_loss'] = loss.item()

            # Add cross-frame RKD distance loss if enabled
            if rkd_distance_criterion is not None:
                rkd_dist_loss, rkd_dist_details = rkd_distance_criterion(teacher_output, student_output)
                if 'base_loss' not in loss_details:
                    loss_details['base_loss'] = loss.item()
                loss_details['rkd_dist_weighted'] = (args.rkd_distance_weight * rkd_dist_loss).item()
                loss = loss + args.rkd_distance_weight * rkd_dist_loss
                loss_details.update(rkd_dist_details)
                loss_details['total_loss'] = loss.item()

            # Add output-level distillation loss if enabled
            if use_output_loss:
                output_loss, output_details = output_criterion(teacher_preds, student_preds)
                if 'base_loss' not in loss_details:
                    loss_details['base_loss'] = loss.item()
                loss_details['output_weighted'] = (args.output_weight * output_loss).item()
                loss = loss + args.output_weight * output_loss
                loss_details.update(output_details)
                loss_details['total_loss'] = loss.item()

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
    rkd_criterion: Optional[nn.Module] = None,
    rkd_distance_criterion: Optional[nn.Module] = None,
    output_criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    student.eval()
    teacher.eval()

    device = args.device
    total_loss = 0.0
    total_details = {}
    num_batches = 0
    use_output_loss = output_criterion is not None and args.output_weight > 0.0

    for batch in val_loader:
        teacher_images = batch['teacher_images'].to(device)
        student_images = batch['student_images'].to(device)

        with autocast(enabled=args.use_amp):
            if use_output_loss:
                teacher_output, teacher_preds = teacher.forward_with_preds(teacher_images)
                student_output, student_preds = student.forward_with_preds(student_images)
            else:
                teacher_output = teacher(teacher_images)
                student_output = student(student_images)

            loss, loss_details = criterion(teacher_output, student_output)

            if rkd_criterion is not None:
                rkd_loss, rkd_details = rkd_criterion(teacher_output, student_output)
                loss_details['base_loss'] = loss.item()
                loss_details['rkd_weighted'] = (args.rkd_weight * rkd_loss).item()
                loss = loss + args.rkd_weight * rkd_loss
                loss_details.update(rkd_details)
                loss_details['total_loss'] = loss.item()

            if rkd_distance_criterion is not None:
                rkd_dist_loss, rkd_dist_details = rkd_distance_criterion(teacher_output, student_output)
                if 'base_loss' not in loss_details:
                    loss_details['base_loss'] = loss.item()
                loss_details['rkd_dist_weighted'] = (args.rkd_distance_weight * rkd_dist_loss).item()
                loss = loss + args.rkd_distance_weight * rkd_dist_loss
                loss_details.update(rkd_dist_details)
                loss_details['total_loss'] = loss.item()

            if use_output_loss:
                output_loss, output_details = output_criterion(teacher_preds, student_preds)
                if 'base_loss' not in loss_details:
                    loss_details['base_loss'] = loss.item()
                loss_details['output_weighted'] = (args.output_weight * output_loss).item()
                loss = loss + args.output_weight * output_loss
                loss_details.update(output_details)
                loss_details['total_loss'] = loss.item()

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
    parser.add_argument('--stride_sampling', action='store_true',
                        help='Enable stride-based anchor sampling (random anchor + stride window)')
    parser.add_argument('--stride', type=int, default=2,
                        help='File-list stride for consecutive views (default: 2)')
    parser.add_argument('--paired_sampling', action='store_true',
                        help='Enable paired anchor+companion frame sampling')
    parser.add_argument('--paired_gap', type=int, default=3,
                        help='Gap between anchor and companion in sorted file list (default: 3)')
    parser.add_argument('--fixed_subset_seed', type=int, default=None,
                        help='Pre-compute benchmark-style frame subset per scene using this seed')
    parser.add_argument('--fixed_subset_max_frames', type=int, default=100,
                        help='Max frames for fixed subset (default: 100)')
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
    parser.add_argument('--warmup_ratio', type=float, default=None,
                        help='Warmup ratio (0-1) of total steps. Overrides --warmup_steps if set.')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                        help='Minimum LR for cosine scheduler')
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
    parser.add_argument('--patch_huber_cosine', action='store_true',
                        help='DA3-style patch Huber + cosine loss on full token')
    parser.add_argument('--patch_huber_weight', type=float, default=1.0,
                        help='Weight for patch Huber term')
    parser.add_argument('--patch_huber_cos_weight', type=float, default=2.0,
                        help='Weight for patch cosine term')
    parser.add_argument('--patch_huber_delta', type=float, default=1.0,
                        help='Huber delta (beta) for patch Huber loss')
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

    # Cross-frame RKD loss arguments
    parser.add_argument('--cross_frame_rkd', action='store_true',
                        help='Enable cross-frame RKD angle-wise loss (additive to main loss)')
    parser.add_argument('--rkd_weight', type=float, default=1.0,
                        help='Weight for cross-frame RKD loss')
    parser.add_argument('--rkd_topk', type=int, default=4,
                        help='Top-K similar patches from extra frames')
    parser.add_argument('--rkd_num_ref_samples', type=int, default=128,
                        help='Number of reference patch positions to sample')
    parser.add_argument('--rkd_num_shared_samples', type=int, default=128,
                        help='Number of shared patch positions to sample')
    parser.add_argument('--rkd_angle1_weight', type=float, default=1.0,
                        help='Weight for angle 1 (vertex=ref)')
    parser.add_argument('--rkd_angle2_weight', type=float, default=1.0,
                        help='Weight for angle 2 (vertex=sim_high)')
    parser.add_argument('--rkd_angle3_weight', type=float, default=1.0,
                        help='Weight for angle 3 (vertex=shared)')
    parser.add_argument('--rkd_shared_chunk_size', type=int, default=64,
                        help='Chunk size for shared positions to bound memory')
    parser.add_argument('--rkd_selection_mode', type=str, default='topk',
                        choices=['topk', 'random', 'mixed'],
                        help='Patch selection mode for RKD: topk, random, mixed (top+bottom)')
    parser.add_argument('--use_rkd_distance', action='store_true',
                        help='Enable RKD distance-wise loss alongside angle-wise RKD')
    parser.add_argument('--rkd_distance_weight', type=float, default=1.0,
                        help='Weight for RKD distance-wise loss')
    parser.add_argument('--rkd_distance_chunk_size', type=int, default=16,
                        help='Chunk size for RKD distance computation')
    parser.add_argument('--rkd_distance_type', type=str, default='l2', choices=['l2', 'cosine'],
                        help='Distance metric setting (kept for parity with DA3)')
    parser.add_argument('--rkd_normalize_distance', action='store_true', default=True,
                        help='Normalize distances by mean (parity with DA3)')
    parser.add_argument('--rkd_no_normalize_distance', action='store_true',
                        help='Disable distance normalization')
    parser.add_argument('--rkd_d1_weight', type=float, default=1.0,
                        help='Weight for distance edge d1 (ref-shared)')
    parser.add_argument('--rkd_d2_weight', type=float, default=1.0,
                        help='Weight for distance edge d2 (ref-sim_high)')
    parser.add_argument('--rkd_d3_weight', type=float, default=1.0,
                        help='Weight for distance edge d3 (shared-sim_high)')
    parser.add_argument('--rkd_distance_temperature', type=float, default=1.0,
                        help='Temperature for RKD distance softmax KL mode')
    parser.add_argument('--rkd_distance_mode', type=str, default='kl', choices=['kl', 'huber'],
                        help='Distance RKD mode: softmax KL or direct Huber')
    parser.add_argument('--rkd_distance_huber_beta', type=float, default=0.5,
                        help='Huber beta when rkd_distance_mode=huber')

    # Output-level distillation loss arguments
    parser.add_argument('--output_weight', type=float, default=0.0,
                        help='Weight for output-level distillation loss (0 = disabled)')
    parser.add_argument('--output_camera_weight', type=float, default=5.0,
                        help='Weight for camera pose_enc L1 loss within output loss')
    parser.add_argument('--output_depth_weight', type=float, default=1.0,
                        help='Weight for depth L1 loss within output loss')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (small batch, few steps)')

    args = parser.parse_args()

    if args.rkd_no_normalize_distance:
        args.rkd_normalize_distance = False

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
    if args.patch_huber_cosine:
        print("\nUsing VGGTPatchHuberCosineLoss (DA3-style patch Huber + cosine)")
        criterion = VGGTPatchHuberCosineLoss(
            student_frame_indices=student_indices,
            target_layers=args.output_layers,
            huber_weight=args.patch_huber_weight,
            cos_weight=args.patch_huber_cos_weight,
            delta=args.patch_huber_delta,
        )
    elif args.all_token_softmax_kl_cosine:
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

    # Create optional cross-frame RKD loss
    rkd_criterion = None
    if args.cross_frame_rkd:
        print(f"\nAdding VGGTCrossFrameRKDAngleLoss (weight={args.rkd_weight})")
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
            selection_mode=args.rkd_selection_mode,
        )
    rkd_distance_criterion = None
    if args.use_rkd_distance:
        print(
            f"\nAdding VGGTCrossFrameRKDDistanceLoss "
            f"(weight={args.rkd_distance_weight}, mode={args.rkd_distance_mode})"
        )
        rkd_distance_criterion = VGGTCrossFrameRKDDistanceLoss(
            student_frame_indices=student_indices,
            num_teacher_views=args.num_views,
            target_layer=target_layer,
            topk=args.rkd_topk,
            num_ref_samples=args.rkd_num_ref_samples,
            num_shared_samples=args.rkd_num_shared_samples,
            d1_weight=args.rkd_d1_weight,
            d2_weight=args.rkd_d2_weight,
            d3_weight=args.rkd_d3_weight,
            shared_chunk_size=args.rkd_shared_chunk_size,
            distance_chunk_size=args.rkd_distance_chunk_size,
            distance_type=args.rkd_distance_type,
            normalize_distance=args.rkd_normalize_distance,
            temperature=args.rkd_distance_temperature,
            distance_mode=args.rkd_distance_mode,
            huber_beta=args.rkd_distance_huber_beta,
            selection_mode=args.rkd_selection_mode,
        )

    # Create optional output-level distillation loss
    output_criterion = None
    if args.output_weight > 0.0:
        print(f"\nAdding VGGTOutputDistillLoss (weight={args.output_weight}, cam={args.output_camera_weight}, depth={args.output_depth_weight})")
        output_criterion = VGGTOutputDistillLoss(
            student_frame_indices=student_indices,
            camera_weight=args.output_camera_weight,
            depth_weight=args.output_depth_weight,
        )

    # Create optimizer (only LoRA params)
    optimizer = torch.optim.AdamW(
        student.get_trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    num_training_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps
    if args.warmup_ratio is not None:
        warmup_steps = int(num_training_steps * args.warmup_ratio)
        print(f"Warmup ratio {args.warmup_ratio} -> {warmup_steps} steps (of {num_training_steps} total)")
    scheduler = get_lr_scheduler(
        optimizer,
        args.lr_scheduler,
        num_training_steps,
        warmup_steps,
        len(train_loader),
        eta_min=args.eta_min,
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

        # Update dataset epoch for per-epoch sampling diversity
        train_loader.dataset.current_epoch = epoch

        # Train
        global_step = train_epoch(
            teacher, student, train_loader, criterion,
            optimizer, scheduler, scaler,
            epoch, global_step, args, writer,
            rkd_criterion=rkd_criterion,
            rkd_distance_criterion=rkd_distance_criterion,
            output_criterion=output_criterion,
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
