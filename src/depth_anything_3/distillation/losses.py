"""
Distillation Loss Functions for DA3 Knowledge Distillation.

This module provides loss functions for distilling knowledge from a teacher
model (8 views) to a student model (4 views).

Key classes:
- LocalTokenSoftmaxKLCosineLoss: Local token per-token channel softmax KL + cosine
- GlobalTokenSoftmaxKLCosineLoss: Global token per-token channel softmax KL + cosine
- CombinedTokenSoftmaxKLCosineLoss: Combined local+global softmax KL + cosine
- AllTokenSoftmaxKLCosineLoss: Full 3072-dim token softmax KL + cosine (no split)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from depth_anything_3.distillation.models import DistillationOutput


class LocalTokenSoftmaxKLCosineLoss(nn.Module):
    """Local token per-token channel softmax KL + cosine loss."""

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        kl_weight: float = 1.0,
        cos_weight: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.kl_weight = kl_weight
        self.cos_weight = cos_weight
        print(f"LocalTokenSoftmaxKLCosineLoss: layer={self.target_layer}, kl_w={kl_weight}, cos_w={cos_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_local = teacher_output.local_features[self.target_layer]
        student_local = student_output.local_features[self.target_layer]
        teacher_local_selected = teacher_local[:, self.student_frame_indices, :, :]

        # Per-token channel softmax
        teacher_sm = torch.softmax(teacher_local_selected, dim=-1)
        student_sm = torch.softmax(student_local, dim=-1)

        kl = (teacher_sm * (torch.log(teacher_sm + 1e-8) - torch.log(student_sm + 1e-8))).sum(dim=-1).mean()
        # Cosine per token
        teacher_norm = torch.nn.functional.normalize(teacher_sm, dim=-1)
        student_norm = torch.nn.functional.normalize(student_sm, dim=-1)
        cos = (teacher_norm * student_norm).sum(dim=-1).mean()

        loss = self.kl_weight * kl + self.cos_weight * (1.0 - cos)

        return loss, {
            'local_softmax_kl': kl.item(),
            'local_softmax_cos': cos.item(),
            'local_softmax_total': loss.item(),
            'total_loss': loss.item(),
        }


class GlobalTokenSoftmaxKLCosineLoss(nn.Module):
    """Global token per-token channel softmax KL + cosine loss."""

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        kl_weight: float = 1.0,
        cos_weight: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.kl_weight = kl_weight
        self.cos_weight = cos_weight
        print(f"GlobalTokenSoftmaxKLCosineLoss: layer={self.target_layer}, kl_w={kl_weight}, cos_w={cos_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_global = teacher_output.global_features[self.target_layer]
        student_global = student_output.global_features[self.target_layer]
        teacher_global_selected = teacher_global[:, self.student_frame_indices, :, :]

        # Per-token channel softmax
        teacher_sm = torch.softmax(teacher_global_selected, dim=-1)
        student_sm = torch.softmax(student_global, dim=-1)

        kl = (teacher_sm * (torch.log(teacher_sm + 1e-8) - torch.log(student_sm + 1e-8))).sum(dim=-1).mean()
        # Cosine per token
        teacher_norm = torch.nn.functional.normalize(teacher_sm, dim=-1)
        student_norm = torch.nn.functional.normalize(student_sm, dim=-1)
        cos = (teacher_norm * student_norm).sum(dim=-1).mean()

        loss = self.kl_weight * kl + self.cos_weight * (1.0 - cos)

        return loss, {
            'global_softmax_kl': kl.item(),
            'global_softmax_cos': cos.item(),
            'global_softmax_total': loss.item(),
            'total_loss': loss.item(),
        }


class CombinedTokenSoftmaxKLCosineLoss(nn.Module):
    """
    Combined local+global per-token channel softmax KL + cosine loss.
    Each branch has its own weights; total is the sum of both totals.
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        local_kl_weight: float = 1.0,
        local_cos_weight: float = 1.0,
        global_kl_weight: float = 1.0,
        global_cos_weight: float = 1.0,
    ):
        super().__init__()
        self.local_loss = LocalTokenSoftmaxKLCosineLoss(
            student_frame_indices=student_frame_indices,
            kl_weight=local_kl_weight,
            cos_weight=local_cos_weight,
        )
        self.global_loss = GlobalTokenSoftmaxKLCosineLoss(
            student_frame_indices=student_frame_indices,
            kl_weight=global_kl_weight,
            cos_weight=global_cos_weight,
        )
        print(
            "CombinedTokenSoftmaxKLCosineLoss:",
            f"local kl_w={local_kl_weight}, cos_w={local_cos_weight}; "
            f"global kl_w={global_kl_weight}, cos_w={global_cos_weight}",
        )

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_local, det_local = self.local_loss(teacher_output, student_output)
        loss_global, det_global = self.global_loss(teacher_output, student_output)
        loss = loss_local + loss_global
        details = {
            **det_local,
            **det_global,
            'combined_softmax_total': loss.item(),
            'total_loss': loss.item(),
        }
        return loss, details


class AllTokenSoftmaxKLCosineLoss(nn.Module):
    """
    Full-token (3072-dim) channel softmax KL + cosine loss without splitting.

    This treats the concatenated [local, global] token as one vector instead of
    computing separate losses on the two halves. Useful when we want gradients
    to flow jointly across both parts of the concatenated feature.
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        kl_weight: float = 1.0,
        cos_weight: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.kl_weight = kl_weight
        self.cos_weight = cos_weight
        print(f"AllTokenSoftmaxKLCosineLoss: layer={self.target_layer}, "
              f"kl_w={kl_weight}, cos_w={cos_weight}")
        print("  Uses full 3072-dim token (no local/global split)")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_all = teacher_output.layer_features[self.target_layer]  # [B, 8/16, P, 3072]
        student_all = student_output.layer_features[self.target_layer]  # [B, 4, P, 3072]
        teacher_selected = teacher_all[:, self.student_frame_indices, :, :]

        # Per-token channel softmax over the full 3072 dims
        teacher_sm = torch.softmax(teacher_selected, dim=-1)
        student_sm = torch.softmax(student_all, dim=-1)

        kl = (teacher_sm * (torch.log(teacher_sm + 1e-8) - torch.log(student_sm + 1e-8))).sum(dim=-1).mean()

        # Cosine on the same full token (use normalized raw tokens for direction)
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
