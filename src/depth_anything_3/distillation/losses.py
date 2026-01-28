"""
Distillation Loss Functions for DA3 Knowledge Distillation.

This module provides loss functions for distilling knowledge from a teacher
model (8 views) to a student model (4 views). The losses operate on:
- Global features only (second half of concatenated output when cat_token=True)
- Camera tokens (learned per-view representations)

Key classes:
- RobustRegressionLoss: Robust feature matching loss
- CosineDistillLoss: Cosine similarity loss
- DA3DistillationLoss: Combined loss for DA3 distillation
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_3.distillation.models import DistillationOutput


class RobustRegressionLoss(nn.Module):
    """
    Generalized Robust Loss for feature distillation.

    Based on https://arxiv.org/abs/1701.03077, adapted for high-dimensional
    features. More robust to outliers than MSE.

    Args:
        alpha: Shape parameter controlling robustness (lower = more robust)
        scaling_c: Scale parameter for transition between quadratic and robust
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: float = 0.5,
        scaling_c: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.scaling_c = scaling_c
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute robust regression loss.

        Args:
            pred: Predicted features [B, S, P, C] or [B, S, C]
            target: Target features (same shape as pred)

        Returns:
            Scalar loss value
        """
        assert pred.shape == target.shape, (
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )

        # Compute scaled squared error, summed over feature dimension
        error_scaled = torch.sum(
            ((pred - target) / self.scaling_c) ** 2,
            dim=-1
        )

        # Apply robust loss formula
        alpha = self.alpha
        robust_loss = (abs(alpha - 2) / alpha) * (
            torch.pow((error_scaled / abs(alpha - 2)) + 1, alpha / 2) - 1
        )

        # Apply reduction
        if self.reduction == "none":
            return robust_loss
        elif self.reduction == "sum":
            return robust_loss.sum()
        elif self.reduction == "mean":
            return robust_loss.mean() if robust_loss.numel() > 0 else robust_loss.new_zeros(())
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class CosineDistillLoss(nn.Module):
    """
    Cosine similarity loss for feature distillation.

    Maximizes cosine similarity between teacher and student features.

    Args:
        temperature: Temperature scaling factor
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss.

        Args:
            pred: Predicted features [B, S, P, C] or [B, S, C]
            target: Target features (same shape as pred)

        Returns:
            Scalar loss value (higher similarity = lower loss)
        """
        assert pred.shape == target.shape, (
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )

        # Normalize features along the channel dimension
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)

        # Compute cosine similarity
        cos_sim = (pred_norm * target_norm).sum(dim=-1)

        # Loss: 1 - similarity (want to maximize similarity)
        loss = (1 - cos_sim) / self.temperature

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean() if loss.numel() > 0 else loss.new_zeros(())
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class DA3DistillationLoss(nn.Module):
    """
    Combined distillation loss for DA3 knowledge distillation.

    This loss:
    1. Operates on GLOBAL features (robust + cosine loss) at all DPT input layers
    2. Distills camera tokens separately
    3. Applies robust/cosine losses at layers 19, 27, 33, 39 (DPT inputs)
    4. Applies KL loss at layers 33, 39 (cross-view layers)
    5. Matches student frames to corresponding teacher frames

    Args:
        output_layers: Layer indices to extract features from (default: [19, 27, 33, 39])
        student_frame_indices: Which teacher frames correspond to student
        feature_loss_weights: Weights for 'robust', 'cosine', 'kl' losses
        camera_token_weight: Weight for camera token loss
        robust_alpha: Alpha for robust loss
        robust_scaling_c: Scaling for robust loss
        cosine_temperature: Temperature for cosine loss
        kl_temperature: Temperature for KL loss
        robust_cosine_layers: Layers to apply robust/cosine losses (default: [19, 27, 33, 39])
        kl_layers: Layers to apply KL loss (default: [33, 39])
        distill_local_features: Whether to distill local features (default: False)
    """

    def __init__(
        self,
        output_layers: List[int] = None,
        student_frame_indices: List[int] = None,
        feature_loss_weights: Dict[str, float] = None,
        camera_token_weight: float = 0.0,  # Disabled by default
        robust_alpha: float = 0.5,
        robust_scaling_c: float = 0.25,
        cosine_temperature: float = 0.1,
        kl_temperature: float = 0.07,
        robust_cosine_layers: List[int] = None,
        kl_layers: List[int] = None,
        distill_local_features: bool = False,  # Disabled - local features may hurt pose
    ):
        super().__init__()

        # Output layers must match DPT input layers for Giant: 19, 27, 33, 39
        self.output_layers = output_layers or [19, 27, 33, 39]

        # Layers for each loss type
        self.robust_cosine_layers = robust_cosine_layers or [33, 39]  # Only later layers
        self.kl_layers = kl_layers or [33, 39]  # KL only at later layers (cross-view)

        # Student frame indices in teacher
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]

        # Feature loss weights - KL enabled for pose preservation
        self.feature_loss_weights = feature_loss_weights or {
            'robust': 1.0,
            'cosine': 1.0,
            'kl': 1.0,  # Enabled for pose preservation
        }

        # Camera token weight
        self.camera_token_weight = camera_token_weight

        # Whether to distill local features
        self.distill_local_features = distill_local_features

        # Initialize sub-losses
        self.robust_loss = RobustRegressionLoss(
            alpha=robust_alpha,
            scaling_c=robust_scaling_c,
            reduction='mean',
        )

        self.cosine_loss = CosineDistillLoss(
            temperature=cosine_temperature,
            reduction='mean',
        )

        # KL loss parameters
        self.kl_temperature = max(kl_temperature, 0.01)
        self.kl_max_logit = 50.0

        print(f"DA3DistillationLoss initialized:")
        print(f"  Output layers: {self.output_layers}")
        print(f"  Robust/Cosine layers: {self.robust_cosine_layers}")
        print(f"  KL layers: {self.kl_layers}")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  Feature loss weights: {self.feature_loss_weights}")
        print(f"  Camera token weight: {self.camera_token_weight}")
        print(f"  Distill local features: {self.distill_local_features}")

    def _compute_kl_loss(
        self,
        teacher_global: torch.Tensor,
        student_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss with patch-to-all-patches similarity.

        Formula: L_KL = (1/|P|) * sum_p D_KL(softmax(T_i^(p) · T_j / τ) || softmax(S_i^(p) · T_j / τ))

        Where T_j contains ALL patches from ALL teacher frames.

        Args:
            teacher_global: [B, 8, P, C] teacher global features (all 8 views)
            student_global: [B, 4, P, C] student global features

        Returns:
            Scalar KL loss value
        """
        B, S_teacher, P, C = teacher_global.shape
        _, S_student, _, _ = student_global.shape

        # Normalize each patch independently along channel dimension
        teacher_norm = F.normalize(teacher_global, p=2, dim=-1)  # [B, 8, P, C]
        student_norm = F.normalize(student_global, p=2, dim=-1)  # [B, 4, P, C]

        # Flatten teacher patches: [B, 8*P, C] - all patches from all frames
        T_j = teacher_norm.reshape(B, S_teacher * P, C)  # [B, 8*P, C]

        total_kl = torch.tensor(0.0, device=teacher_global.device, dtype=teacher_global.dtype)

        for student_idx, teacher_idx in enumerate(self.student_frame_indices):
            # T_i^(p): Teacher patch features for frame i: [B, P, C]
            T_i_p = teacher_norm[:, teacher_idx, :, :]
            # S_i^(p): Student patch features for frame i: [B, P, C]
            S_i_p = student_norm[:, student_idx, :, :]

            # Compute T_i^(p) · T_j: similarity of each query patch to ALL teacher patches
            # [B, P, C] @ [B, C, 8*P] -> [B, P, 8*P]
            teacher_sim = torch.bmm(T_i_p, T_j.transpose(1, 2))  # [B, P, 8*P]
            student_sim = torch.bmm(S_i_p, T_j.transpose(1, 2))  # [B, P, 8*P]

            # Exclude self-similarity (patches in frame i)
            # Frame i occupies indices [teacher_idx*P : (teacher_idx+1)*P]
            start_idx = teacher_idx * P
            end_idx = (teacher_idx + 1) * P
            teacher_sim[:, :, start_idx:end_idx] = -1e9
            student_sim[:, :, start_idx:end_idx] = -1e9

            # Scale by temperature
            teacher_logits = teacher_sim / self.kl_temperature
            student_logits = student_sim / self.kl_temperature

            # Clamp for numerical stability
            teacher_logits = torch.clamp(teacher_logits, -self.kl_max_logit, self.kl_max_logit)
            student_logits = torch.clamp(student_logits, -self.kl_max_logit, self.kl_max_logit)

            # Softmax to get distributions over all teacher patches
            teacher_dist = F.softmax(teacher_logits, dim=-1)  # [B, P, 8*P]
            student_log_dist = F.log_softmax(student_logits, dim=-1)  # [B, P, 8*P]

            # KL divergence: sum over target patches dimension, then mean over query patches
            kl_per_patch = F.kl_div(student_log_dist, teacher_dist, reduction='none').sum(dim=-1)  # [B, P]

            # Average over patches (1/|P|) and batch
            kl_loss = kl_per_patch.mean()

            # Clamp to ensure non-negative
            kl_loss = torch.clamp(kl_loss, min=0)

            total_kl = total_kl + kl_loss

        # Average over student frames
        avg_kl = total_kl / len(self.student_frame_indices)
        return avg_kl

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.

        Args:
            teacher_output: DistillationOutput from teacher (8 views)
            student_output: DistillationOutput from student (4 views)

        Returns:
            total_loss: Scalar loss value
            loss_details: Dict with individual loss components
        """
        device = next(iter(teacher_output.global_features.values())).device
        total_loss = torch.tensor(0.0, device=device)
        loss_details = {}

        # Compute robust/cosine loss only at specified layers (default: layer 39 only)
        for layer_idx in self.robust_cosine_layers:
            if layer_idx not in teacher_output.global_features:
                continue
            if layer_idx not in student_output.global_features:
                continue

            # Get global features (NOT local_x)
            teacher_global = teacher_output.global_features[layer_idx]  # [B, 8, P, C]
            student_global = student_output.global_features[layer_idx]  # [B, 4, P, C]

            # Select teacher frames matching student indices
            teacher_selected = teacher_global[:, self.student_frame_indices, :, :]  # [B, 4, P, C]

            # Get camera tokens
            teacher_cam = teacher_output.camera_tokens[layer_idx]  # [B, 8, C]
            student_cam = student_output.camera_tokens[layer_idx]  # [B, 4, C]
            teacher_cam_selected = teacher_cam[:, self.student_frame_indices, :]  # [B, 4, C]

            # Feature loss (global features only)
            feat_robust = self.robust_loss(student_global, teacher_selected)
            feat_cosine = self.cosine_loss(student_global, teacher_selected)

            feat_loss = (
                self.feature_loss_weights['robust'] * feat_robust +
                self.feature_loss_weights['cosine'] * feat_cosine
            )

            # Camera token loss
            cam_robust = self.robust_loss(student_cam, teacher_cam_selected)
            cam_cosine = self.cosine_loss(student_cam, teacher_cam_selected)

            cam_loss = (
                self.feature_loss_weights['robust'] * cam_robust +
                self.feature_loss_weights['cosine'] * cam_cosine
            )

            # Combined layer loss
            layer_loss = feat_loss + self.camera_token_weight * cam_loss
            total_loss = total_loss + layer_loss

            # Record details
            loss_details[f'layer_{layer_idx}_feat_robust'] = feat_robust.item()
            loss_details[f'layer_{layer_idx}_feat_cosine'] = feat_cosine.item()
            loss_details[f'layer_{layer_idx}_cam_robust'] = cam_robust.item()
            loss_details[f'layer_{layer_idx}_cam_cosine'] = cam_cosine.item()
            loss_details[f'layer_{layer_idx}_robust_cosine_total'] = layer_loss.item()

            # Local feature loss (for layers 0-12 gradients via local_x)
            if self.distill_local_features:
                # Get local features (first half of concatenated output)
                teacher_local = teacher_output.local_features[layer_idx]  # [B, 8, P, C]
                student_local = student_output.local_features[layer_idx]  # [B, 4, P, C]

                # Select teacher frames matching student indices
                teacher_local_selected = teacher_local[:, self.student_frame_indices, :, :]  # [B, 4, P, C]

                # Local feature loss (robust + cosine)
                local_robust = self.robust_loss(student_local, teacher_local_selected)
                local_cosine = self.cosine_loss(student_local, teacher_local_selected)

                local_loss = (
                    self.feature_loss_weights['robust'] * local_robust +
                    self.feature_loss_weights['cosine'] * local_cosine
                )

                total_loss = total_loss + local_loss

                loss_details[f'layer_{layer_idx}_local_robust'] = local_robust.item()
                loss_details[f'layer_{layer_idx}_local_cosine'] = local_cosine.item()
                loss_details[f'layer_{layer_idx}_local_total'] = local_loss.item()

        # Compute KL loss at specified layers (default: layers 33 and 39)
        kl_total = torch.tensor(0.0, device=device)
        for layer_idx in self.kl_layers:
            if layer_idx not in teacher_output.global_features:
                continue
            if layer_idx not in student_output.global_features:
                continue

            teacher_global = teacher_output.global_features[layer_idx]  # [B, 8, P, C]
            student_global = student_output.global_features[layer_idx]  # [B, 4, P, C]

            kl_loss = self._compute_kl_loss(teacher_global, student_global)
            kl_total = kl_total + kl_loss

            loss_details[f'layer_{layer_idx}_kl'] = kl_loss.item()

        # Average KL over layers
        if len(self.kl_layers) > 0:
            kl_total = kl_total / len(self.kl_layers)

        # Add weighted KL to total
        total_loss = total_loss + self.feature_loss_weights['kl'] * kl_total

        loss_details['kl_total'] = kl_total.item()
        loss_details['total_loss'] = total_loss.item()

        return total_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"output_layers={self.output_layers}, "
            f"robust_cosine_layers={self.robust_cosine_layers}, "
            f"kl_layers={self.kl_layers}, "
            f"student_frame_indices={self.student_frame_indices}"
        )


class CameraTokenOnlyLoss(nn.Module):
    """
    Camera token only distillation loss for pose estimation improvement.

    This loss focuses exclusively on making student camera tokens match
    teacher camera tokens, without any spatial feature matching.

    Uses only layer 39 (the last layer) which is what the camera decoder receives.
    Uses the full 3072-dim token [local_x[:,:,0], x[:,:,0]] that goes to cam_dec.

    Combines two loss components:
    1. Smooth L1 (Huber) loss: Matches magnitude, robust to outliers
    2. Cosine loss: Matches direction in feature space

    Reference view is fixed to view 0 for both teacher and student to ensure
    consistent camera token semantics.

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
        smooth_l1_beta: Threshold for Smooth L1 switching from MSE to L1 (default 1.0)
        smooth_l1_weight: Weight for Smooth L1 loss component (default 1.0)
        cosine_weight: Weight for cosine loss component (default 1.0)
    """

    def __init__(
        self,
        output_layers: List[int] = None,  # Kept for API compatibility, but ignored
        student_frame_indices: List[int] = None,
        smooth_l1_beta: float = 1.0,
        smooth_l1_weight: float = 1.0,
        cosine_weight: float = 1.0,
        # Legacy parameters kept for API compatibility
        robust_alpha: float = 0.5,
        robust_scaling_c: float = 0.5,
        cosine_temperature: float = 0.1,
        robust_weight: float = 1.0,
    ):
        super().__init__()
        # Only use layer 39 - this is what camera decoder receives
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.smooth_l1_beta = smooth_l1_beta
        self.smooth_l1_weight = smooth_l1_weight
        self.cosine_weight = cosine_weight

        print(f"CameraTokenOnlyLoss initialized:")
        print(f"  Target layer: {self.target_layer} (camera decoder input)")
        print(f"  Token dimension: 3072 (full [local_x, x] concatenation)")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  Smooth L1: beta={smooth_l1_beta}, weight={smooth_l1_weight}")
        print(f"  Cosine: weight={cosine_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute camera token only distillation loss.

        Combines Smooth L1 (magnitude matching) and Cosine (direction matching).

        Args:
            teacher_output: DistillationOutput from teacher (8/16 views)
            student_output: DistillationOutput from student (4 views)

        Returns:
            total_loss: Scalar loss value
            loss_details: Dict with individual loss components
        """
        # Get full camera tokens from layer 39 (what camera decoder receives)
        # Shape: [B, S, 3072] = [local_x[:,:,0], x[:,:,0]]
        teacher_cam_full = teacher_output.camera_tokens_full[self.target_layer]  # [B, 8/16, 3072]
        student_cam_full = student_output.camera_tokens_full[self.target_layer]  # [B, 4, 3072]

        # Select teacher frames matching student indices
        teacher_cam_selected = teacher_cam_full[:, self.student_frame_indices, :]  # [B, 4, 3072]

        # Smooth L1 loss (magnitude matching, robust to outliers)
        smooth_l1_loss = F.smooth_l1_loss(
            student_cam_full, teacher_cam_selected, beta=self.smooth_l1_beta
        )

        # Cosine loss (direction matching)
        # Compute cosine similarity: [B, 4]
        cos_sim = F.cosine_similarity(student_cam_full, teacher_cam_selected, dim=-1)
        # Loss = 1 - similarity (want to maximize similarity)
        cosine_loss = (1 - cos_sim).mean()

        # Combined loss
        total_loss = (
            self.smooth_l1_weight * smooth_l1_loss +
            self.cosine_weight * cosine_loss
        )

        loss_details = {
            'layer_39_smooth_l1': smooth_l1_loss.item(),
            'layer_39_cosine': cosine_loss.item(),
            'layer_39_cos_sim': cos_sim.mean().item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"target_layer={self.target_layer}, "
            f"student_frame_indices={self.student_frame_indices}, "
            f"smooth_l1_weight={self.smooth_l1_weight}, "
            f"cosine_weight={self.cosine_weight}"
        )


class CameraTokenMSELoss(nn.Module):
    """
    Simple MSE loss for camera token distillation.

    Uses only layer 39 (camera decoder input) with plain MSE loss.
    This is the simplest possible loss for camera token matching.

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,  # Kept for API compatibility, ignored
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]

        print(f"CameraTokenMSELoss initialized:")
        print(f"  Target layer: {self.target_layer} (camera decoder input)")
        print(f"  Token dimension: 3072 (full [local_x, x] concatenation)")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  Loss: MSE only")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute simple MSE loss between teacher and student camera tokens.

        Args:
            teacher_output: DistillationOutput from teacher (8/16 views)
            student_output: DistillationOutput from student (4 views)

        Returns:
            total_loss: Scalar loss value
            loss_details: Dict with loss components
        """
        # Get full camera tokens from layer 39 (what camera decoder receives)
        # Shape: [B, S, 3072] = [local_x[:,:,0], x[:,:,0]]
        teacher_cam = teacher_output.camera_tokens_full[self.target_layer]  # [B, 8/16, 3072]
        student_cam = student_output.camera_tokens_full[self.target_layer]  # [B, 4, 3072]

        # Select teacher frames matching student indices
        teacher_selected = teacher_cam[:, self.student_frame_indices, :]  # [B, 4, 3072]

        # Simple MSE loss
        mse_loss = F.mse_loss(student_cam, teacher_selected)

        loss_details = {
            'mse_loss': mse_loss.item(),
            'total_loss': mse_loss.item(),
        }

        return mse_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"target_layer={self.target_layer}, "
            f"student_frame_indices={self.student_frame_indices}"
        )


class CameraTokenCosineLoss(nn.Module):
    """
    Cosine similarity only loss for camera token distillation.

    Uses only layer 39 (camera decoder input) with pure cosine similarity loss.
    This focuses on matching the direction/orientation of camera tokens in the
    feature space, ignoring magnitude differences.

    Loss = 1 - cosine_similarity(student, teacher)

    This is useful when we want the student to learn the same "direction" in
    feature space as the teacher, which may be more important for pose estimation
    than matching exact magnitudes.

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
        temperature: Temperature scaling factor (lower = sharper gradients)
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,  # Kept for API compatibility, ignored
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.temperature = temperature

        print(f"CameraTokenCosineLoss initialized:")
        print(f"  Target layer: {self.target_layer} (camera decoder input)")
        print(f"  Token dimension: 3072 (full [local_x, x] concatenation)")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  Temperature: {temperature}")
        print(f"  Loss: Cosine similarity only (no MSE)")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cosine similarity loss between teacher and student camera tokens.

        Args:
            teacher_output: DistillationOutput from teacher (8/16 views)
            student_output: DistillationOutput from student (4 views)

        Returns:
            total_loss: Scalar loss value
            loss_details: Dict with loss components
        """
        # Get full camera tokens from layer 39 (what camera decoder receives)
        # Shape: [B, S, 3072] = [local_x[:,:,0], x[:,:,0]]
        teacher_cam = teacher_output.camera_tokens_full[self.target_layer]  # [B, 8/16, 3072]
        student_cam = student_output.camera_tokens_full[self.target_layer]  # [B, 4, 3072]

        # Select teacher frames matching student indices
        teacher_selected = teacher_cam[:, self.student_frame_indices, :]  # [B, 4, 3072]

        # Normalize features along the feature dimension
        student_norm = F.normalize(student_cam, p=2, dim=-1)  # [B, 4, 3072]
        teacher_norm = F.normalize(teacher_selected, p=2, dim=-1)  # [B, 4, 3072]

        # Compute cosine similarity: element-wise product then sum over feature dim
        cos_sim = (student_norm * teacher_norm).sum(dim=-1)  # [B, 4]

        # Loss = (1 - cosine_similarity) / temperature
        # When cos_sim = 1 (perfect match), loss = 0
        # When cos_sim = -1 (opposite), loss = 2/temperature
        cosine_loss = (1 - cos_sim) / self.temperature  # [B, 4]
        cosine_loss = cosine_loss.mean()

        loss_details = {
            'cosine_loss': cosine_loss.item(),
            'cos_sim_mean': cos_sim.mean().item(),
            'cos_sim_min': cos_sim.min().item(),
            'cos_sim_max': cos_sim.max().item(),
            'total_loss': cosine_loss.item(),
        }

        return cosine_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"target_layer={self.target_layer}, "
            f"student_frame_indices={self.student_frame_indices}, "
            f"temperature={self.temperature}"
        )


class PatchFeatureCosineLoss(nn.Module):
    """
    Cosine similarity loss for patch feature distillation.

    Computes cosine similarity between teacher and student patch features
    at layer 39 (camera decoder input). Shape: [B, S, P, 3072].

    Loss = 1 - cosine_similarity(student, teacher)

    This focuses on matching the direction/orientation of patch features,
    which captures semantic similarity while being scale-invariant.

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
        temperature: Temperature scaling factor (lower = sharper gradients)
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,  # Kept for API compatibility, ignored
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.temperature = temperature

        print(f"PatchFeatureCosineLoss initialized:")
        print(f"  Target layer: {self.target_layer} (camera decoder input)")
        print(f"  Feature shape: [B, S, P, 3072] (local + global)")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  Temperature: {temperature}")
        print(f"  Loss: Cosine similarity only")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cosine similarity loss between teacher and student patch features.

        Args:
            teacher_output: DistillationOutput from teacher (8/16 views)
            student_output: DistillationOutput from student (4 views)

        Returns:
            total_loss: Scalar loss value
            loss_details: Dict with loss components
        """
        # Get patch features from layer 39
        # Shape: [B, S, P, 3072] = [local_features, global_features]
        teacher_feats = teacher_output.layer_features[self.target_layer]  # [B, 8/16, P, 3072]
        student_feats = student_output.layer_features[self.target_layer]  # [B, 4, P, 3072]

        # Select teacher frames matching student indices
        teacher_selected = teacher_feats[:, self.student_frame_indices, :, :]  # [B, 4, P, 3072]

        # Compute cosine similarity per patch: [B, S, P]
        cos_sim = F.cosine_similarity(student_feats, teacher_selected, dim=-1)

        # Loss = (1 - cosine_similarity) / temperature
        cosine_loss = (1 - cos_sim) / self.temperature
        cosine_loss = cosine_loss.mean()

        # Compute detailed statistics
        loss_details = {
            'patch_cosine_loss': cosine_loss.item(),
            'patch_cos_sim_mean': cos_sim.mean().item(),
            'patch_cos_sim_std': cos_sim.std().item(),
            'patch_cos_sim_min': cos_sim.min().item(),
            'patch_cos_sim_max': cos_sim.max().item(),
            'total_loss': cosine_loss.item(),
        }

        return cosine_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"target_layer={self.target_layer}, "
            f"student_frame_indices={self.student_frame_indices}, "
            f"temperature={self.temperature}"
        )


class CombinedPatchCameraCosineLoss(nn.Module):
    """
    Combined loss: Patch Feature Cosine + Camera Token Cosine.

    This loss combines:
    1. Patch feature cosine loss [B, S, P, 3072] - spatial feature alignment
    2. Camera token cosine loss [B, S, 3072] - view-level alignment

    Both use cosine similarity to focus on feature direction rather than magnitude.

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
        patch_weight: Weight for patch feature loss (default 1.0)
        camera_weight: Weight for camera token loss (default 0.1)
        temperature: Temperature scaling factor
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,  # Kept for API compatibility, ignored
        patch_weight: float = 1.0,
        camera_weight: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.patch_weight = patch_weight
        self.camera_weight = camera_weight
        self.temperature = temperature

        print(f"CombinedPatchCameraCosineLoss initialized:")
        print(f"  Target layer: {self.target_layer} (camera decoder input)")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  Patch weight: {patch_weight}")
        print(f"  Camera weight: {camera_weight}")
        print(f"  Temperature: {temperature}")
        print(f"  Total loss = {patch_weight}*patch_cosine + {camera_weight}*camera_cosine")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined patch + camera cosine loss.

        Args:
            teacher_output: DistillationOutput from teacher (8/16 views)
            student_output: DistillationOutput from student (4 views)

        Returns:
            total_loss: Scalar loss value
            loss_details: Dict with loss components
        """
        # ============ Patch Feature Loss [B, S, P, 3072] ============
        teacher_feats = teacher_output.layer_features[self.target_layer]
        student_feats = student_output.layer_features[self.target_layer]
        teacher_feats_selected = teacher_feats[:, self.student_frame_indices, :, :]

        # Cosine similarity per patch
        patch_cos_sim = F.cosine_similarity(student_feats, teacher_feats_selected, dim=-1)
        patch_loss = (1 - patch_cos_sim) / self.temperature
        patch_loss = patch_loss.mean()

        # ============ Camera Token Loss [B, S, 3072] ============
        teacher_cam = teacher_output.camera_tokens_full[self.target_layer]
        student_cam = student_output.camera_tokens_full[self.target_layer]
        teacher_cam_selected = teacher_cam[:, self.student_frame_indices, :]

        # Cosine similarity per view
        cam_cos_sim = F.cosine_similarity(student_cam, teacher_cam_selected, dim=-1)
        cam_loss = (1 - cam_cos_sim) / self.temperature
        cam_loss = cam_loss.mean()

        # ============ Combined Loss ============
        total_loss = self.patch_weight * patch_loss + self.camera_weight * cam_loss

        loss_details = {
            # Patch feature stats
            'patch_cosine_loss': patch_loss.item(),
            'patch_cos_sim_mean': patch_cos_sim.mean().item(),
            'patch_cos_sim_std': patch_cos_sim.std().item(),
            'patch_cos_sim_min': patch_cos_sim.min().item(),
            'patch_cos_sim_max': patch_cos_sim.max().item(),
            # Camera token stats
            'camera_cosine_loss': cam_loss.item(),
            'camera_cos_sim_mean': cam_cos_sim.mean().item(),
            'camera_cos_sim_min': cam_cos_sim.min().item(),
            'camera_cos_sim_max': cam_cos_sim.max().item(),
            # Total
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"target_layer={self.target_layer}, "
            f"student_frame_indices={self.student_frame_indices}, "
            f"patch_weight={self.patch_weight}, "
            f"camera_weight={self.camera_weight}, "
            f"temperature={self.temperature}"
        )


class GlobalFeatureMSELoss(nn.Module):
    """
    Option A: Global features only MSE loss.

    loss = MSE(student_global_39, teacher_global_39)

    Uses only the global (second half) of the concatenated features at layer 39.
    This is conservative as it only distills the cross-view attention output.

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,  # Kept for API compatibility, ignored
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]

        print(f"GlobalFeatureMSELoss (Option A) initialized:")
        print(f"  Target layer: {self.target_layer}")
        print(f"  Feature: Global only (second half, 1536-dim)")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  Loss: MSE on global features only")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute MSE loss on global features only.

        Args:
            teacher_output: DistillationOutput from teacher (8/16 views)
            student_output: DistillationOutput from student (4 views)

        Returns:
            total_loss: Scalar loss value
            loss_details: Dict with loss components
        """
        # Get global features from layer 39 (second half of concatenated output)
        # Shape: [B, S, P, 1536]
        teacher_global = teacher_output.global_features[self.target_layer]  # [B, 8/16, P, 1536]
        student_global = student_output.global_features[self.target_layer]  # [B, 4, P, 1536]

        # Select teacher frames matching student indices
        teacher_selected = teacher_global[:, self.student_frame_indices, :, :]  # [B, 4, P, 1536]

        # MSE loss on global features
        mse_loss = F.mse_loss(student_global, teacher_selected)

        loss_details = {
            'global_mse_loss': mse_loss.item(),
            'total_loss': mse_loss.item(),
        }

        return mse_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"target_layer={self.target_layer}, "
            f"student_frame_indices={self.student_frame_indices}"
        )


class GlobalFeatureCosineLoss(nn.Module):
    """
    Cosine similarity loss on global features only.

    loss = 1 - cosine_similarity(student_global_39, teacher_global_39)

    Uses only the global (second half) of the concatenated features at layer 39.
    This is scale-invariant and focuses on feature direction matching.

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
        temperature: Temperature scaling factor
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,  # Kept for API compatibility, ignored
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.temperature = temperature

        print(f"GlobalFeatureCosineLoss initialized:")
        print(f"  Target layer: {self.target_layer}")
        print(f"  Feature: Global only (second half, 1536-dim)")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  Temperature: {temperature}")
        print(f"  Loss: Cosine similarity on global features only")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cosine similarity loss on global features only.
        """
        # Get global features from layer 39 (second half of concatenated output)
        # Shape: [B, S, P, 1536]
        teacher_global = teacher_output.global_features[self.target_layer]
        student_global = student_output.global_features[self.target_layer]

        # Select teacher frames matching student indices
        teacher_selected = teacher_global[:, self.student_frame_indices, :, :]

        # Compute cosine similarity per patch: [B, S, P]
        cos_sim = F.cosine_similarity(student_global, teacher_selected, dim=-1)

        # Loss = (1 - cosine_similarity) / temperature
        cosine_loss = (1 - cos_sim) / self.temperature
        cosine_loss = cosine_loss.mean()

        loss_details = {
            'global_cosine_loss': cosine_loss.item(),
            'global_cos_sim_mean': cos_sim.mean().item(),
            'global_cos_sim_std': cos_sim.std().item(),
            'global_cos_sim_min': cos_sim.min().item(),
            'global_cos_sim_max': cos_sim.max().item(),
            'total_loss': cosine_loss.item(),
        }

        return cosine_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"target_layer={self.target_layer}, "
            f"student_frame_indices={self.student_frame_indices}, "
            f"temperature={self.temperature}"
        )


class GlobalFeatureMSECosineLoss(nn.Module):
    """
    Combined MSE + Cosine loss on global features only.

    loss = mse_weight * MSE(student_global, teacher_global) + cosine_weight * (1 - cosine_sim)

    This combines magnitude matching (MSE) with direction matching (Cosine).

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
        mse_weight: Weight for MSE loss (default 1.0)
        cosine_weight: Weight for cosine loss (default 1.0)
        temperature: Temperature for cosine loss
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        mse_weight: float = 1.0,
        cosine_weight: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.temperature = temperature

        print(f"GlobalFeatureMSECosineLoss initialized:")
        print(f"  Target layer: {self.target_layer}")
        print(f"  Feature: Global only (second half, 1536-dim)")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  MSE weight: {mse_weight}")
        print(f"  Cosine weight: {cosine_weight}")
        print(f"  Temperature: {temperature}")
        print(f"  Loss = {mse_weight}*MSE + {cosine_weight}*Cosine")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined MSE + Cosine loss on global features.
        """
        # Get global features from layer 39
        teacher_global = teacher_output.global_features[self.target_layer]
        student_global = student_output.global_features[self.target_layer]

        # Select teacher frames matching student indices
        teacher_selected = teacher_global[:, self.student_frame_indices, :, :]

        # MSE loss
        mse_loss = F.mse_loss(student_global, teacher_selected)

        # Cosine loss
        cos_sim = F.cosine_similarity(student_global, teacher_selected, dim=-1)
        cosine_loss = (1 - cos_sim) / self.temperature
        cosine_loss = cosine_loss.mean()

        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss

        loss_details = {
            'global_mse_loss': mse_loss.item(),
            'global_cosine_loss': cosine_loss.item(),
            'global_cos_sim_mean': cos_sim.mean().item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"target_layer={self.target_layer}, "
            f"student_frame_indices={self.student_frame_indices}, "
            f"mse_weight={self.mse_weight}, "
            f"cosine_weight={self.cosine_weight}"
        )


class LocalGlobalFeatureMSELoss(nn.Module):
    """
    Option B: Both local + global features with weighting.

    loss = w1 * MSE(student_local_39, teacher_local_39) + w2 * MSE(student_global_39, teacher_global_39)

    Uses both local (first half) and global (second half) of the concatenated features.
    Typically w1 < w2 to prevent local features from dominating.

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
        local_weight: Weight for local feature MSE (w1, default 0.5)
        global_weight: Weight for global feature MSE (w2, default 1.0)
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,  # Kept for API compatibility, ignored
        local_weight: float = 0.5,
        global_weight: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.local_weight = local_weight
        self.global_weight = global_weight

        print(f"LocalGlobalFeatureMSELoss (Option B) initialized:")
        print(f"  Target layer: {self.target_layer}")
        print(f"  Features: Local (1536-dim) + Global (1536-dim)")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  Local weight (w1): {local_weight}")
        print(f"  Global weight (w2): {global_weight}")
        print(f"  Loss = {local_weight}*MSE(local) + {global_weight}*MSE(global)")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted MSE loss on local + global features.

        Args:
            teacher_output: DistillationOutput from teacher (8/16 views)
            student_output: DistillationOutput from student (4 views)

        Returns:
            total_loss: Scalar loss value
            loss_details: Dict with loss components
        """
        # Get local features (first half of concatenated output)
        # Shape: [B, S, P, 1536]
        teacher_local = teacher_output.local_features[self.target_layer]  # [B, 8/16, P, 1536]
        student_local = student_output.local_features[self.target_layer]  # [B, 4, P, 1536]

        # Get global features (second half of concatenated output)
        teacher_global = teacher_output.global_features[self.target_layer]  # [B, 8/16, P, 1536]
        student_global = student_output.global_features[self.target_layer]  # [B, 4, P, 1536]

        # Select teacher frames matching student indices
        teacher_local_selected = teacher_local[:, self.student_frame_indices, :, :]
        teacher_global_selected = teacher_global[:, self.student_frame_indices, :, :]

        # MSE losses
        local_mse = F.mse_loss(student_local, teacher_local_selected)
        global_mse = F.mse_loss(student_global, teacher_global_selected)

        # Weighted combination
        total_loss = self.local_weight * local_mse + self.global_weight * global_mse

        loss_details = {
            'local_mse_loss': local_mse.item(),
            'global_mse_loss': global_mse.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"target_layer={self.target_layer}, "
            f"student_frame_indices={self.student_frame_indices}, "
            f"local_weight={self.local_weight}, "
            f"global_weight={self.global_weight}"
        )


class CosineWithOptionLoss(nn.Module):
    """
    Cosine loss + Option A/B/C combination.

    Combines the current cosine loss with one of:
    - Option A: Global features MSE
    - Option B: Local + Global features MSE
    - Option C: Camera token MSE

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
        option: 'A', 'B', or 'C'
        cosine_weight: Weight for cosine loss (default 1.0)
        option_weight: Weight for option loss (default 1.0)
        local_weight: For option B, weight for local MSE (default 0.5)
        global_weight: For option B, weight for global MSE (default 1.0)
        temperature: Temperature for cosine loss
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        option: str = 'A',
        cosine_weight: float = 1.0,
        option_weight: float = 1.0,
        local_weight: float = 0.5,
        global_weight: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.option = option.upper()
        self.cosine_weight = cosine_weight
        self.option_weight = option_weight
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.temperature = temperature

        print(f"CosineWithOptionLoss (Cosine + Option {self.option}) initialized:")
        print(f"  Target layer: {self.target_layer}")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  Cosine weight: {cosine_weight}")
        print(f"  Option {self.option} weight: {option_weight}")
        if self.option == 'B':
            print(f"  Local weight (w1): {local_weight}")
            print(f"  Global weight (w2): {global_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined cosine + option loss."""
        loss_details = {}

        # ============ Camera Token Cosine Loss ============
        teacher_cam = teacher_output.camera_tokens_full[self.target_layer]
        student_cam = student_output.camera_tokens_full[self.target_layer]
        teacher_cam_selected = teacher_cam[:, self.student_frame_indices, :]

        # Cosine similarity
        cos_sim = F.cosine_similarity(student_cam, teacher_cam_selected, dim=-1)
        cosine_loss = (1 - cos_sim) / self.temperature
        cosine_loss = cosine_loss.mean()

        loss_details['cosine_loss'] = cosine_loss.item()
        loss_details['cos_sim_mean'] = cos_sim.mean().item()

        # ============ Option Loss ============
        if self.option == 'A':
            # Global features MSE only
            teacher_global = teacher_output.global_features[self.target_layer]
            student_global = student_output.global_features[self.target_layer]
            teacher_selected = teacher_global[:, self.student_frame_indices, :, :]
            option_loss = F.mse_loss(student_global, teacher_selected)
            loss_details['global_mse_loss'] = option_loss.item()

        elif self.option == 'B':
            # Local + Global features MSE
            teacher_local = teacher_output.local_features[self.target_layer]
            student_local = student_output.local_features[self.target_layer]
            teacher_global = teacher_output.global_features[self.target_layer]
            student_global = student_output.global_features[self.target_layer]

            teacher_local_selected = teacher_local[:, self.student_frame_indices, :, :]
            teacher_global_selected = teacher_global[:, self.student_frame_indices, :, :]

            local_mse = F.mse_loss(student_local, teacher_local_selected)
            global_mse = F.mse_loss(student_global, teacher_global_selected)
            option_loss = self.local_weight * local_mse + self.global_weight * global_mse

            loss_details['local_mse_loss'] = local_mse.item()
            loss_details['global_mse_loss'] = global_mse.item()

        elif self.option == 'C':
            # Camera token MSE
            option_loss = F.mse_loss(student_cam, teacher_cam_selected)
            loss_details['camera_mse_loss'] = option_loss.item()

        else:
            raise ValueError(f"Unknown option: {self.option}")

        loss_details['option_loss'] = option_loss.item()

        # ============ Combined Loss ============
        total_loss = self.cosine_weight * cosine_loss + self.option_weight * option_loss

        loss_details['total_loss'] = total_loss.item()

        return total_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"target_layer={self.target_layer}, "
            f"option={self.option}, "
            f"cosine_weight={self.cosine_weight}, "
            f"option_weight={self.option_weight}"
        )


class LocalTokenCosineLoss(nn.Module):
    """Local token cosine loss only (no global features)."""

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.temperature = temperature
        print(f"LocalTokenCosineLoss initialized: layer={self.target_layer}, temp={temperature}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_local = teacher_output.local_features[self.target_layer]
        student_local = student_output.local_features[self.target_layer]
        teacher_local_selected = teacher_local[:, self.student_frame_indices, :, :]

        cos_sim = F.cosine_similarity(student_local, teacher_local_selected, dim=-1)
        loss = ((1 - cos_sim) / self.temperature).mean()

        return loss, {
            'local_cosine_loss': loss.item(),
            'local_cos_sim_mean': cos_sim.mean().item(),
            'total_loss': loss.item(),
        }


class LocalTokenMSELoss(nn.Module):
    """Local token MSE loss only (no global features)."""

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        print(f"LocalTokenMSELoss initialized: layer={self.target_layer}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_local = teacher_output.local_features[self.target_layer]
        student_local = student_output.local_features[self.target_layer]
        teacher_local_selected = teacher_local[:, self.student_frame_indices, :, :]

        loss = F.mse_loss(student_local, teacher_local_selected)

        return loss, {
            'local_mse_loss': loss.item(),
            'total_loss': loss.item(),
        }


class LocalTokenNormMSELoss(nn.Module):
    """Local token MSE loss with L2 normalization."""

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        print(f"LocalTokenNormMSELoss initialized: layer={self.target_layer}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_local = teacher_output.local_features[self.target_layer]
        student_local = student_output.local_features[self.target_layer]
        teacher_local_selected = teacher_local[:, self.student_frame_indices, :, :]

        # L2 normalize before MSE
        teacher_norm = F.normalize(teacher_local_selected, dim=-1)
        student_norm = F.normalize(student_local, dim=-1)
        loss = F.mse_loss(student_norm, teacher_norm)

        return loss, {
            'local_norm_mse_loss': loss.item(),
            'total_loss': loss.item(),
        }


class LocalTokenNormMSECosineLoss(nn.Module):
    """Local token L2-normalized MSE + Cosine loss."""

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        mse_weight: float = 1.0,
        cosine_weight: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.temperature = temperature
        print(f"LocalTokenNormMSECosineLoss: layer={self.target_layer}, "
              f"mse_w={mse_weight}, cos_w={cosine_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_local = teacher_output.local_features[self.target_layer]
        student_local = student_output.local_features[self.target_layer]
        teacher_local_selected = teacher_local[:, self.student_frame_indices, :, :]

        # L2 normalized MSE
        teacher_norm = F.normalize(teacher_local_selected, dim=-1)
        student_norm = F.normalize(student_local, dim=-1)
        mse_loss = F.mse_loss(student_norm, teacher_norm)

        # Cosine loss
        cos_sim = F.cosine_similarity(student_local, teacher_local_selected, dim=-1)
        cosine_loss = ((1 - cos_sim) / self.temperature).mean()

        total = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss

        return total, {
            'local_norm_mse_loss': mse_loss.item(),
            'local_cosine_loss': cosine_loss.item(),
            'local_cos_sim_mean': cos_sim.mean().item(),
            'total_loss': total.item(),
        }


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


class LocalTokenNormKLCosineLoss(nn.Module):
    """Local token KL (channel softmax) + cosine on L2-normalized tokens."""

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
        print(f"LocalTokenNormKLCosineLoss: layer={self.target_layer}, kl_w={kl_weight}, cos_w={cos_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_local = teacher_output.local_features[self.target_layer]
        student_local = student_output.local_features[self.target_layer]
        teacher_local_selected = teacher_local[:, self.student_frame_indices, :, :]

        # L2 normalize tokens
        t_norm = torch.nn.functional.normalize(teacher_local_selected, dim=-1)
        s_norm = torch.nn.functional.normalize(student_local, dim=-1)

        # KL on channel softmax of normalized tokens
        t_sm = torch.softmax(t_norm, dim=-1)
        s_sm = torch.softmax(s_norm, dim=-1)
        kl = (t_sm * (torch.log(t_sm + 1e-8) - torch.log(s_sm + 1e-8))).sum(dim=-1).mean()

        # Cosine on normalized tokens
        cos = (t_norm * s_norm).sum(dim=-1).mean()

        loss = self.kl_weight * kl + self.cos_weight * (1.0 - cos)

        return loss, {
            'local_norm_kl': kl.item(),
            'local_norm_cos': cos.item(),
            'local_norm_kl_cos_total': loss.item(),
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


class LocalTokenRobustLoss(nn.Module):
    """Local token Robust Regression loss only."""

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        alpha: float = 0.5,
        scaling_c: float = 0.25,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.robust_loss = RobustRegressionLoss(alpha=alpha, scaling_c=scaling_c)
        print(f"LocalTokenRobustLoss: layer={self.target_layer}, alpha={alpha}, c={scaling_c}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_local = teacher_output.local_features[self.target_layer]
        student_local = student_output.local_features[self.target_layer]
        teacher_local_selected = teacher_local[:, self.student_frame_indices, :, :]

        loss = self.robust_loss(student_local, teacher_local_selected)

        return loss, {
            'local_robust_loss': loss.item(),
            'total_loss': loss.item(),
        }


class LocalTokenNormRobustLoss(nn.Module):
    """Local token L2-normalized Robust Regression loss."""

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        alpha: float = 0.5,
        scaling_c: float = 0.25,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.robust_loss = RobustRegressionLoss(alpha=alpha, scaling_c=scaling_c)
        print(f"LocalTokenNormRobustLoss: layer={self.target_layer}, alpha={alpha}, c={scaling_c}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_local = teacher_output.local_features[self.target_layer]
        student_local = student_output.local_features[self.target_layer]
        teacher_local_selected = teacher_local[:, self.student_frame_indices, :, :]

        # L2 normalize before robust loss
        teacher_norm = F.normalize(teacher_local_selected, dim=-1)
        student_norm = F.normalize(student_local, dim=-1)
        loss = self.robust_loss(student_norm, teacher_norm)

        return loss, {
            'local_norm_robust_loss': loss.item(),
            'total_loss': loss.item(),
        }


class LocalGlobalRobustCosineLoss(nn.Module):
    """
    Combined loss:
    - Local: L2-norm Robust + Cosine
    - Global: Robust + Cosine
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        local_norm_robust_weight: float = 1.0,
        local_cosine_weight: float = 1.0,
        global_robust_weight: float = 1.0,
        global_cosine_weight: float = 1.0,
        alpha: float = 0.5,
        scaling_c: float = 0.25,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.local_norm_robust_weight = local_norm_robust_weight
        self.local_cosine_weight = local_cosine_weight
        self.global_robust_weight = global_robust_weight
        self.global_cosine_weight = global_cosine_weight
        self.temperature = temperature
        self.robust_loss = RobustRegressionLoss(alpha=alpha, scaling_c=scaling_c)
        print(f"LocalGlobalRobustCosineLoss: layer={self.target_layer}")
        print(f"  Local: norm_robust_w={local_norm_robust_weight}, cosine_w={local_cosine_weight}")
        print(f"  Global: robust_w={global_robust_weight}, cosine_w={global_cosine_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Get features
        teacher_local = teacher_output.local_features[self.target_layer]
        student_local = student_output.local_features[self.target_layer]
        teacher_global = teacher_output.global_features[self.target_layer]
        student_global = student_output.global_features[self.target_layer]

        teacher_local_sel = teacher_local[:, self.student_frame_indices, :, :]
        teacher_global_sel = teacher_global[:, self.student_frame_indices, :, :]

        # Local: L2-norm Robust
        teacher_local_norm = F.normalize(teacher_local_sel, dim=-1)
        student_local_norm = F.normalize(student_local, dim=-1)
        local_robust = self.robust_loss(student_local_norm, teacher_local_norm)

        # Local: Cosine
        local_cos_sim = F.cosine_similarity(student_local, teacher_local_sel, dim=-1)
        local_cosine = ((1 - local_cos_sim) / self.temperature).mean()

        # Global: Robust (no norm)
        global_robust = self.robust_loss(student_global, teacher_global_sel)

        # Global: Cosine
        global_cos_sim = F.cosine_similarity(student_global, teacher_global_sel, dim=-1)
        global_cosine = ((1 - global_cos_sim) / self.temperature).mean()

        total = (
            self.local_norm_robust_weight * local_robust +
            self.local_cosine_weight * local_cosine +
            self.global_robust_weight * global_robust +
            self.global_cosine_weight * global_cosine
        )

        return total, {
            'local_norm_robust_loss': local_robust.item(),
            'local_cosine_loss': local_cosine.item(),
            'local_cos_sim_mean': local_cos_sim.mean().item(),
            'global_robust_loss': global_robust.item(),
            'global_cosine_loss': global_cosine.item(),
            'global_cos_sim_mean': global_cos_sim.mean().item(),
            'total_loss': total.item(),
        }


class LocalGlobalNormRobustCosineLoss(nn.Module):
    """
    Combined loss with BOTH local and global normalized:
    - Local: L2-norm Robust + Cosine
    - Global: L2-norm Robust + Cosine
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        local_norm_robust_weight: float = 1.0,
        local_cosine_weight: float = 1.0,
        global_norm_robust_weight: float = 1.0,
        global_cosine_weight: float = 1.0,
        alpha: float = 0.5,
        scaling_c: float = 0.25,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.local_norm_robust_weight = local_norm_robust_weight
        self.local_cosine_weight = local_cosine_weight
        self.global_norm_robust_weight = global_norm_robust_weight
        self.global_cosine_weight = global_cosine_weight
        self.temperature = temperature
        self.robust_loss = RobustRegressionLoss(alpha=alpha, scaling_c=scaling_c)
        print(f"LocalGlobalNormRobustCosineLoss: layer={self.target_layer}")
        print(f"  Local: norm_robust_w={local_norm_robust_weight}, cosine_w={local_cosine_weight}")
        print(f"  Global: norm_robust_w={global_norm_robust_weight}, cosine_w={global_cosine_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Get features
        teacher_local = teacher_output.local_features[self.target_layer]
        student_local = student_output.local_features[self.target_layer]
        teacher_global = teacher_output.global_features[self.target_layer]
        student_global = student_output.global_features[self.target_layer]

        teacher_local_sel = teacher_local[:, self.student_frame_indices, :, :]
        teacher_global_sel = teacher_global[:, self.student_frame_indices, :, :]

        # Local: L2-norm Robust
        teacher_local_norm = F.normalize(teacher_local_sel, dim=-1)
        student_local_norm = F.normalize(student_local, dim=-1)
        local_robust = self.robust_loss(student_local_norm, teacher_local_norm)

        # Local: Cosine
        local_cos_sim = F.cosine_similarity(student_local, teacher_local_sel, dim=-1)
        local_cosine = ((1 - local_cos_sim) / self.temperature).mean()

        # Global: L2-norm Robust (NOW NORMALIZED)
        teacher_global_norm = F.normalize(teacher_global_sel, dim=-1)
        student_global_norm = F.normalize(student_global, dim=-1)
        global_robust = self.robust_loss(student_global_norm, teacher_global_norm)

        # Global: Cosine
        global_cos_sim = F.cosine_similarity(student_global, teacher_global_sel, dim=-1)
        global_cosine = ((1 - global_cos_sim) / self.temperature).mean()

        total = (
            self.local_norm_robust_weight * local_robust +
            self.local_cosine_weight * local_cosine +
            self.global_norm_robust_weight * global_robust +
            self.global_cosine_weight * global_cosine
        )

        return total, {
            'local_norm_robust_loss': local_robust.item(),
            'local_cosine_loss': local_cosine.item(),
            'local_cos_sim_mean': local_cos_sim.mean().item(),
            'global_norm_robust_loss': global_robust.item(),
            'global_cosine_loss': global_cosine.item(),
            'global_cos_sim_mean': global_cos_sim.mean().item(),
            'total_loss': total.item(),
        }


class MultiLayerNormRobustCosineLoss(nn.Module):
    """
    Multi-layer distillation loss applying LocalGlobalNormRobustCosineLoss to multiple layers.

    Applies the same loss structure to layer 39 (primary) and layer 33 (secondary),
    with configurable weight for the secondary layer.
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        local_norm_robust_weight: float = 3.0,
        local_cosine_weight: float = 1.0,
        global_norm_robust_weight: float = 1.0,
        global_cosine_weight: float = 1.0,
        layer33_weight: float = 0.1,
        alpha: float = 0.5,
        scaling_c: float = 0.25,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.layers = [39, 33]
        self.layer_weights = {39: 1.0, 33: layer33_weight}
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.local_norm_robust_weight = local_norm_robust_weight
        self.local_cosine_weight = local_cosine_weight
        self.global_norm_robust_weight = global_norm_robust_weight
        self.global_cosine_weight = global_cosine_weight
        self.temperature = temperature
        self.robust_loss = RobustRegressionLoss(alpha=alpha, scaling_c=scaling_c)
        print(f"MultiLayerNormRobustCosineLoss: layers={self.layers}, weights={self.layer_weights}")
        print(f"  Local: norm_robust_w={local_norm_robust_weight}, cosine_w={local_cosine_weight}")
        print(f"  Global: norm_robust_w={global_norm_robust_weight}, cosine_w={global_cosine_weight}")

    def _compute_layer_loss(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
        layer: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a single layer."""
        teacher_local = teacher_output.local_features[layer]
        student_local = student_output.local_features[layer]
        teacher_global = teacher_output.global_features[layer]
        student_global = student_output.global_features[layer]

        teacher_local_sel = teacher_local[:, self.student_frame_indices, :, :]
        teacher_global_sel = teacher_global[:, self.student_frame_indices, :, :]

        # Local: L2-norm Robust
        teacher_local_norm = F.normalize(teacher_local_sel, dim=-1)
        student_local_norm = F.normalize(student_local, dim=-1)
        local_robust = self.robust_loss(student_local_norm, teacher_local_norm)

        # Local: Cosine
        local_cos_sim = F.cosine_similarity(student_local, teacher_local_sel, dim=-1)
        local_cosine = ((1 - local_cos_sim) / self.temperature).mean()

        # Global: L2-norm Robust
        teacher_global_norm = F.normalize(teacher_global_sel, dim=-1)
        student_global_norm = F.normalize(student_global, dim=-1)
        global_robust = self.robust_loss(student_global_norm, teacher_global_norm)

        # Global: Cosine
        global_cos_sim = F.cosine_similarity(student_global, teacher_global_sel, dim=-1)
        global_cosine = ((1 - global_cos_sim) / self.temperature).mean()

        total = (
            self.local_norm_robust_weight * local_robust +
            self.local_cosine_weight * local_cosine +
            self.global_norm_robust_weight * global_robust +
            self.global_cosine_weight * global_cosine
        )

        metrics = {
            f'L{layer}_local_robust': local_robust.item(),
            f'L{layer}_local_cosine': local_cosine.item(),
            f'L{layer}_global_robust': global_robust.item(),
            f'L{layer}_global_cosine': global_cosine.item(),
            f'L{layer}_total': total.item(),
        }
        return total, metrics

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        all_metrics = {}

        for layer in self.layers:
            layer_loss, layer_metrics = self._compute_layer_loss(
                teacher_output, student_output, layer
            )
            weight = self.layer_weights[layer]
            total_loss = total_loss + weight * layer_loss
            all_metrics.update(layer_metrics)
            all_metrics[f'L{layer}_weighted'] = weight * layer_loss.item()

        all_metrics['total_loss'] = total_loss.item()
        return total_loss, all_metrics


class SeparateCosineOptionALoss(nn.Module):
    """
    Separate Local/Global Cosine + Global MSE loss.

    This loss computes cosine similarity SEPARATELY on local and global features,
    then adds MSE loss on global features only.

    loss = local_cosine_weight * (1 - cos_sim(local))
         + global_cosine_weight * (1 - cos_sim(global))
         + global_mse_weight * MSE(global)

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
        local_cosine_weight: Weight for local cosine loss (default 1.0)
        global_cosine_weight: Weight for global cosine loss (default 1.0)
        global_mse_weight: Weight for global MSE loss (default 1.0)
        temperature: Temperature for cosine loss
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        local_cosine_weight: float = 1.0,
        global_cosine_weight: float = 1.0,
        global_mse_weight: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.local_cosine_weight = local_cosine_weight
        self.global_cosine_weight = global_cosine_weight
        self.global_mse_weight = global_mse_weight
        self.temperature = temperature

        print(f"SeparateCosineOptionALoss initialized:")
        print(f"  Target layer: {self.target_layer}")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  Local cosine weight: {local_cosine_weight}")
        print(f"  Global cosine weight: {global_cosine_weight}")
        print(f"  Global MSE weight: {global_mse_weight}")
        print(f"  Temperature: {temperature}")
        print(f"  Loss = {local_cosine_weight}*local_cosine + {global_cosine_weight}*global_cosine + {global_mse_weight}*global_mse")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute separate local/global cosine + global MSE loss."""
        loss_details = {}

        # Get local features (first half, 1536-dim)
        teacher_local = teacher_output.local_features[self.target_layer]  # [B, 8, P, 1536]
        student_local = student_output.local_features[self.target_layer]  # [B, 4, P, 1536]
        teacher_local_selected = teacher_local[:, self.student_frame_indices, :, :]

        # Get global features (second half, 1536-dim)
        teacher_global = teacher_output.global_features[self.target_layer]  # [B, 8, P, 1536]
        student_global = student_output.global_features[self.target_layer]  # [B, 4, P, 1536]
        teacher_global_selected = teacher_global[:, self.student_frame_indices, :, :]

        # ============ Local Cosine Loss (1536-dim) ============
        local_cos_sim = F.cosine_similarity(student_local, teacher_local_selected, dim=-1)  # [B, 4, P]
        local_cosine_loss = (1 - local_cos_sim) / self.temperature
        local_cosine_loss = local_cosine_loss.mean()

        loss_details['local_cosine_loss'] = local_cosine_loss.item()
        loss_details['local_cos_sim_mean'] = local_cos_sim.mean().item()

        # ============ Global Cosine Loss (1536-dim) ============
        global_cos_sim = F.cosine_similarity(student_global, teacher_global_selected, dim=-1)  # [B, 4, P]
        global_cosine_loss = (1 - global_cos_sim) / self.temperature
        global_cosine_loss = global_cosine_loss.mean()

        loss_details['global_cosine_loss'] = global_cosine_loss.item()
        loss_details['global_cos_sim_mean'] = global_cos_sim.mean().item()

        # ============ Global MSE Loss (1536-dim) ============
        global_mse_loss = F.mse_loss(student_global, teacher_global_selected)

        loss_details['global_mse_loss'] = global_mse_loss.item()

        # ============ Combined Loss ============
        total_loss = (
            self.local_cosine_weight * local_cosine_loss +
            self.global_cosine_weight * global_cosine_loss +
            self.global_mse_weight * global_mse_loss
        )

        loss_details['total_loss'] = total_loss.item()

        return total_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"target_layer={self.target_layer}, "
            f"local_cosine_weight={self.local_cosine_weight}, "
            f"global_cosine_weight={self.global_cosine_weight}, "
            f"global_mse_weight={self.global_mse_weight}"
        )


class CrossViewSimilarityDistillationLoss(nn.Module):
    """
    Cross-view similarity distillation loss using global tokens only.

    This loss leverages the extra 4 views from the teacher (8 views) that the student
    doesn't see (4 views). It distills the cross-view similarity patterns from teacher
    to student using only the global features (second half of concatenated tokens, 1536-dim).

    Formula:
        R_teacher = T_shared @ T_extra.T  # [B, 4, 4]
        R_student = S_shared @ T_extra.T  # [B, 4, 4]

        if use_magnitude:
            loss += MSE(R_student, R_teacher)  # Match response strength
        if use_direction:
            loss += (1 - cosine_sim(R_student, R_teacher))  # Match response direction

    Options (2x2 = 4 combinations):
        - normalize_before_mm: Whether to normalize features before matrix multiplication
        - use_magnitude: MSE loss on similarity matrices (match response strength)
        - use_direction: Cosine loss on similarity matrices (match response direction)

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
        normalize_before_mm: Normalize features before computing similarity (default False)
        use_magnitude: Use MSE loss on similarity matrices (default True)
        use_direction: Use cosine loss on similarity matrices (default False)
        magnitude_weight: Weight for magnitude (MSE) loss (default 1.0)
        direction_weight: Weight for direction (cosine) loss (default 1.0)
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        normalize_before_mm: bool = False,
        use_magnitude: bool = True,
        use_direction: bool = False,
        magnitude_weight: float = 1.0,
        direction_weight: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.normalize_before_mm = normalize_before_mm
        self.use_magnitude = use_magnitude
        self.use_direction = use_direction
        self.magnitude_weight = magnitude_weight
        self.direction_weight = direction_weight

        # Compute extra view indices (views not in student)
        all_views = set(range(8))
        self.extra_frame_indices = sorted(list(all_views - set(self.student_frame_indices)))

        print(f"CrossViewSimilarityDistillationLoss initialized:")
        print(f"  Target layer: {self.target_layer}")
        print(f"  Using: Global tokens only (1536-dim, second half of camera_tokens)")
        print(f"  Student frame indices (shared): {self.student_frame_indices}")
        print(f"  Extra frame indices: {self.extra_frame_indices}")
        print(f"  Normalize before mm: {normalize_before_mm}")
        print(f"  Use magnitude (MSE): {use_magnitude}, weight={magnitude_weight}")
        print(f"  Use direction (Cosine): {use_direction}, weight={direction_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cross-view similarity distillation loss.
        """
        # Use global tokens (camera_tokens is the second half, 1536-dim)
        # camera_tokens shape: [B, S, 1536]
        teacher_tokens = teacher_output.camera_tokens[self.target_layer]  # [B, 8, 1536]
        student_tokens = student_output.camera_tokens[self.target_layer]  # [B, 4, 1536]

        # Get shared and extra views from teacher
        T_shared = teacher_tokens[:, self.student_frame_indices, :]  # [B, 4, 1536]
        T_extra = teacher_tokens[:, self.extra_frame_indices, :]  # [B, 4, 1536]
        S_shared = student_tokens  # [B, 4, 1536]

        # Optionally normalize features before matrix multiplication
        if self.normalize_before_mm:
            T_shared = F.normalize(T_shared, dim=-1)
            T_extra = F.normalize(T_extra, dim=-1)
            S_shared = F.normalize(S_shared, dim=-1)
        else:
            # Scale features BEFORE matrix multiplication to prevent overflow
            # Without this, dot products of 1536-dim vectors overflow to inf
            scale = T_shared.shape[-1] ** 0.5  # sqrt(1536) ≈ 39.2
            T_shared = T_shared / scale
            T_extra = T_extra / scale
            S_shared = S_shared / scale

        # Compute cross-view response matrices
        # R[b, i, j] = dot product between shared view i and extra view j
        R_teacher = torch.bmm(T_shared, T_extra.transpose(1, 2))  # [B, 4, 4]
        R_student = torch.bmm(S_shared, T_extra.transpose(1, 2))  # [B, 4, 4]

        loss = torch.tensor(0.0, device=R_teacher.device, dtype=R_teacher.dtype)
        loss_details = {}

        # Magnitude loss (MSE): match response strength
        if self.use_magnitude:
            mag_loss = F.mse_loss(R_student, R_teacher)
            loss = loss + self.magnitude_weight * mag_loss
            loss_details['magnitude_loss'] = mag_loss.item()

        # Direction loss (Cosine): match response direction pattern
        if self.use_direction:
            R_teacher_norm = F.normalize(R_teacher, dim=-1)  # [B, 4, 4]
            R_student_norm = F.normalize(R_student, dim=-1)  # [B, 4, 4]
            # Cosine similarity per row, then average
            cos_sim = (R_teacher_norm * R_student_norm).sum(dim=-1)  # [B, 4]
            dir_loss = (1 - cos_sim).mean()
            loss = loss + self.direction_weight * dir_loss
            loss_details['direction_loss'] = dir_loss.item()
            loss_details['direction_cos_sim'] = cos_sim.mean().item()

        # Compute statistics for logging
        with torch.no_grad():
            R_diff = (R_student - R_teacher).abs()
            loss_details['R_teacher_mean'] = R_teacher.mean().item()
            loss_details['R_student_mean'] = R_student.mean().item()
            loss_details['R_diff_mean'] = R_diff.mean().item()
            loss_details['R_diff_max'] = R_diff.max().item()

        loss_details['total_loss'] = loss.item()

        return loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"target_layer={self.target_layer}, "
            f"normalize_before_mm={self.normalize_before_mm}, "
            f"use_magnitude={self.use_magnitude}, "
            f"use_direction={self.use_direction}"
        )


class CombinedCrossViewLoss(nn.Module):
    """
    Combined loss: Any base loss + Cross-view similarity distillation.

    This allows combining the cross-view similarity loss with other losses
    like cosine, MSE, etc.

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
        base_loss_type: Type of base loss ('cosine', 'mse', 'global_mse', 'global_mse_cosine', 'none')
        cross_view_weight: Weight for cross-view similarity loss
        base_weight: Weight for base loss
        temperature: Temperature for base cosine loss
        normalize_before_mm: Normalize before matrix multiplication in cross-view
        use_magnitude: Use MSE loss in cross-view
        use_direction: Use cosine loss in cross-view
        mse_weight: Weight for MSE in global_mse_cosine (default 1.0)
        cosine_weight: Weight for cosine in global_mse_cosine (default 1.0)
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        base_loss_type: str = 'cosine',
        cross_view_weight: float = 1.0,
        base_weight: float = 1.0,
        temperature: float = 1.0,
        normalize_before_mm: bool = False,
        use_magnitude: bool = True,
        use_direction: bool = False,
        magnitude_weight: float = 1.0,
        direction_weight: float = 1.0,
        mse_weight: float = 1.0,
        cosine_weight: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.base_loss_type = base_loss_type.lower()
        self.cross_view_weight = cross_view_weight
        self.base_weight = base_weight
        self.temperature = temperature
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

        # Initialize cross-view loss with new parameters
        self.cross_view_loss = CrossViewSimilarityDistillationLoss(
            student_frame_indices=student_frame_indices,
            normalize_before_mm=normalize_before_mm,
            use_magnitude=use_magnitude,
            use_direction=use_direction,
            magnitude_weight=magnitude_weight,
            direction_weight=direction_weight,
        )

        print(f"CombinedCrossViewLoss initialized:")
        print(f"  Base loss type: {base_loss_type}")
        print(f"  Base weight: {base_weight}")
        print(f"  Cross-view weight: {cross_view_weight}")
        if base_loss_type.lower() == 'global_mse_cosine':
            print(f"  MSE weight: {mse_weight}")
            print(f"  Cosine weight: {cosine_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined base + cross-view loss."""
        loss_details = {}
        total_loss = torch.tensor(0.0, device=next(iter(teacher_output.global_features.values())).device)

        # Compute base loss
        if self.base_loss_type != 'none':
            teacher_cam = teacher_output.camera_tokens_full[self.target_layer]
            student_cam = student_output.camera_tokens_full[self.target_layer]
            teacher_cam_selected = teacher_cam[:, self.student_frame_indices, :]

            if self.base_loss_type == 'cosine':
                cos_sim = F.cosine_similarity(student_cam, teacher_cam_selected, dim=-1)
                base_loss = (1 - cos_sim) / self.temperature
                base_loss = base_loss.mean()
                loss_details['base_cosine_loss'] = base_loss.item()
                loss_details['cos_sim_mean'] = cos_sim.mean().item()

            elif self.base_loss_type == 'mse':
                base_loss = F.mse_loss(student_cam, teacher_cam_selected)
                loss_details['base_mse_loss'] = base_loss.item()

            elif self.base_loss_type == 'global_mse':
                teacher_global = teacher_output.global_features[self.target_layer]
                student_global = student_output.global_features[self.target_layer]
                teacher_selected = teacher_global[:, self.student_frame_indices, :, :]
                base_loss = F.mse_loss(student_global, teacher_selected)
                loss_details['base_global_mse_loss'] = base_loss.item()

            elif self.base_loss_type == 'global_mse_cosine':
                # Global MSE + Cosine combined (same as GlobalFeatureMSECosineLoss)
                teacher_global = teacher_output.global_features[self.target_layer]
                student_global = student_output.global_features[self.target_layer]
                teacher_selected = teacher_global[:, self.student_frame_indices, :, :]

                # MSE loss on global features
                mse_loss = F.mse_loss(student_global, teacher_selected)
                loss_details['global_mse_loss'] = mse_loss.item()

                # Cosine loss on global features
                cos_sim = F.cosine_similarity(student_global, teacher_selected, dim=-1)
                cosine_loss = (1 - cos_sim) / self.temperature
                cosine_loss = cosine_loss.mean()
                loss_details['global_cosine_loss'] = cosine_loss.item()
                loss_details['global_cos_sim_mean'] = cos_sim.mean().item()

                # Combined base loss
                base_loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss

            else:
                raise ValueError(f"Unknown base_loss_type: {self.base_loss_type}")

            total_loss = total_loss + self.base_weight * base_loss
            loss_details['base_loss'] = base_loss.item()

        # Compute cross-view similarity loss
        cross_view_loss, cv_details = self.cross_view_loss(teacher_output, student_output)
        total_loss = total_loss + self.cross_view_weight * cross_view_loss

        # Merge details
        for k, v in cv_details.items():
            if k != 'total_loss':
                loss_details[k] = v

        loss_details['total_loss'] = total_loss.item()

        return total_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"base_loss_type={self.base_loss_type}, "
            f"base_weight={self.base_weight}, "
            f"cross_view_weight={self.cross_view_weight}"
        )


class AttentionDistillationLoss(nn.Module):
    """
    Attention-guided distillation loss that leverages extra view information.

    This loss distills the teacher's attention distribution from shared patches
    (student's 4 views) to extra patches (teacher's additional 4 views).

    Key idea:
    - Teacher has 8 views: 4 "shared" (indices 0,2,4,6) + 4 "extra" (indices 1,3,5,7)
    - Compute attention from shared patches to extra patches using teacher's features
    - Extract top-k attention indices for each shared patch
    - Compute feature similarity at those top-k positions for both teacher and student
    - MSE loss between teacher and student similarities

    This encourages the student to learn features that would attend similarly to
    the extra views, even though the student never sees those views.

    Args:
        student_frame_indices: Which teacher frames correspond to student frames.
                               Default: [0, 2, 4, 6]
        extra_frame_indices: Which teacher frames are "extra" (not seen by student).
                             Default: [1, 3, 5, 7]
        top_k: Number of top attention positions to use. Default: 10
        temperature: Temperature for computing attention. Default: 0.07
        target_layer: Which layer to compute loss at. Default: 39
        num_heads: Number of attention heads (for averaging). Default: 24 (Giant)
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        extra_frame_indices: List[int] = None,
        top_k: int = 10,
        temperature: float = 0.07,
        target_layer: int = 39,
        num_heads: int = 24,
        robust_alpha: float = 0.5,
        robust_scaling_c: float = 0.25,
        loss_weight: float = 0.2,
    ):
        super().__init__()
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.extra_frame_indices = extra_frame_indices or [1, 3, 5, 7]
        self.top_k = top_k
        self.temperature = temperature
        self.target_layer = target_layer
        self.num_heads = num_heads
        self.robust_alpha = robust_alpha
        self.robust_scaling_c = robust_scaling_c
        self.loss_weight = loss_weight

        print(f"AttentionDistillationLoss initialized:")
        print(f"  Student frame indices: {self.student_frame_indices}")
        print(f"  Extra frame indices: {self.extra_frame_indices}")
        print(f"  Top-k: {top_k}")
        print(f"  Temperature: {temperature}")
        print(f"  Target layer: {target_layer}")
        print(f"  Robust alpha: {robust_alpha}")
        print(f"  Robust scaling_c: {robust_scaling_c}")
        print(f"  Loss weight: {loss_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute attention-guided distillation loss.

        Args:
            teacher_output: DistillationOutput from teacher (8 views)
            student_output: DistillationOutput from student (4 views)

        Returns:
            total_loss: Scalar loss value
            loss_details: Dict with individual loss components
        """
        # Get global features at target layer
        # Teacher: [B, 8, P, C], Student: [B, 4, P, C], C=1536
        teacher_global = teacher_output.global_features[self.target_layer]
        student_global = student_output.global_features[self.target_layer]

        B, S_teacher, P, C = teacher_global.shape

        # Extract shared and extra features from teacher
        teacher_shared = teacher_global[:, self.student_frame_indices, :, :]  # [B, 4, P, C]
        teacher_extra = teacher_global[:, self.extra_frame_indices, :, :]     # [B, 4, P, C]

        # Reshape
        B_shared = teacher_shared.reshape(B, -1, C)  # [B, 4P, C]
        B_extra = teacher_extra.reshape(B, -1, C)    # [B, 4P, C]

        # Normalize for attention computation
        B_shared_norm = F.normalize(B_shared, p=2, dim=-1)
        B_extra_norm = F.normalize(B_extra, p=2, dim=-1)

        # Compute attention: shared -> extra
        attn_logits = torch.bmm(B_shared_norm, B_extra_norm.transpose(-2, -1))
        attn_logits = attn_logits / self.temperature
        teacher_attn = F.softmax(attn_logits, dim=-1)  # [B, 4P, 4P]

        # Get top-k indices
        topk_values, topk_indices = torch.topk(teacher_attn, self.top_k, dim=-1)

        # Student shared features
        student_shared = student_global.reshape(B, -1, C)  # [B, 4P, C]

        # Gather extra features at top-k indices
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, C)
        B_extra_expanded = B_extra.unsqueeze(1).expand(-1, B_shared.shape[1], -1, -1)
        topk_extra_features = torch.gather(B_extra_expanded, dim=2, index=topk_indices_expanded)

        # Compute raw dot product similarities
        R_teacher = torch.bmm(
            B_shared.unsqueeze(2).reshape(B * B_shared.shape[1], 1, C),
            topk_extra_features.reshape(B * B_shared.shape[1], self.top_k, C).transpose(-2, -1)
        ).reshape(B, B_shared.shape[1], self.top_k)

        R_student = torch.bmm(
            student_shared.unsqueeze(2).reshape(B * student_shared.shape[1], 1, C),
            topk_extra_features.reshape(B * student_shared.shape[1], self.top_k, C).transpose(-2, -1)
        ).reshape(B, student_shared.shape[1], self.top_k)

        # Robust loss
        loss = self._robust_loss(R_student, R_teacher) * self.loss_weight

        loss_details = {
            'attn_distill_loss': loss.item(),
            'topk_attn_mean': topk_values.mean().item(),
            'total_loss': loss.item(),
        }

        return loss, loss_details

    def _robust_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute robust regression loss (generalized Charbonnier).

        Args:
            pred: Predicted values [B, 4P, K]
            target: Target values [B, 4P, K]

        Returns:
            Scalar loss value
        """
        # Compute scaled squared error, summed over last dimension (K)
        error_scaled = torch.sum(
            ((pred - target) / self.robust_scaling_c) ** 2,
            dim=-1
        )  # [B, 4P]

        # Apply robust loss formula
        alpha = self.robust_alpha
        robust_loss = (abs(alpha - 2) / alpha) * (
            torch.pow((error_scaled / abs(alpha - 2)) + 1, alpha / 2) - 1
        )

        # Divide by top_k to normalize
        return robust_loss.mean() / self.top_k

    def extra_repr(self) -> str:
        return (
            f"student_frame_indices={self.student_frame_indices}, "
            f"extra_frame_indices={self.extra_frame_indices}, "
            f"top_k={self.top_k}, temperature={self.temperature}"
        )


class LocalRobustWithAttentionLoss(nn.Module):
    """
    Combined loss: LocalGlobalNormRobustCosineLoss + AttentionDistillationLoss.
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        local_norm_robust_weight: float = 3.0,
        local_cosine_weight: float = 1.0,
        global_norm_robust_weight: float = 1.0,
        global_cosine_weight: float = 1.0,
        attn_top_k: int = 10,
        attn_temperature: float = 0.07,
        attn_loss_weight: float = 0.2,
    ):
        super().__init__()
        self.base_loss = LocalGlobalNormRobustCosineLoss(
            student_frame_indices=student_frame_indices,
            local_norm_robust_weight=local_norm_robust_weight,
            local_cosine_weight=local_cosine_weight,
            global_norm_robust_weight=global_norm_robust_weight,
            global_cosine_weight=global_cosine_weight,
        )
        self.attn_loss = AttentionDistillationLoss(
            student_frame_indices=student_frame_indices,
            top_k=attn_top_k,
            temperature=attn_temperature,
            loss_weight=attn_loss_weight,
        )

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Compute base loss
        base_loss, base_details = self.base_loss(teacher_output, student_output)

        # Compute attention loss
        attn_loss, attn_details = self.attn_loss(teacher_output, student_output)

        # Combined loss
        total_loss = base_loss + attn_loss

        # Merge details
        loss_details = base_details.copy()
        for k, v in attn_details.items():
            loss_details[f'attn_{k}'] = v
        loss_details['total_loss'] = total_loss.item()

        return total_loss, loss_details


class CombinedAttentionLoss(nn.Module):
    """
    Combined loss: base feature loss + attention distillation loss.

    This combines a standard feature distillation loss (e.g., cosine, MSE)
    with the attention-guided distillation loss for extra view information.

    Args:
        student_frame_indices: Which teacher frames correspond to student frames
        base_loss_type: Type of base loss ('cosine', 'mse', 'global_mse')
        attn_weight: Weight for attention distillation loss
        base_weight: Weight for base loss
        top_k: Number of top attention positions
        temperature: Temperature for attention computation
        cosine_temperature: Temperature for cosine loss
        target_layer: Which layer to compute loss at
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        base_loss_type: str = 'cosine',
        attn_weight: float = 1.0,
        base_weight: float = 1.0,
        top_k: int = 10,
        temperature: float = 0.07,
        cosine_temperature: float = 0.1,
        target_layer: int = 39,
    ):
        super().__init__()
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.base_loss_type = base_loss_type.lower()
        self.attn_weight = attn_weight
        self.base_weight = base_weight
        self.cosine_temperature = cosine_temperature
        self.target_layer = target_layer

        # Initialize attention distillation loss
        self.attn_loss = AttentionDistillationLoss(
            student_frame_indices=student_frame_indices,
            top_k=top_k,
            temperature=temperature,
            target_layer=target_layer,
        )

        print(f"CombinedAttentionLoss initialized:")
        print(f"  Base loss type: {base_loss_type}")
        print(f"  Base weight: {base_weight}")
        print(f"  Attention weight: {attn_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined base + attention distillation loss."""
        loss_details = {}
        device = next(iter(teacher_output.global_features.values())).device
        total_loss = torch.tensor(0.0, device=device)

        # Compute base loss
        if self.base_loss_type != 'none':
            teacher_global = teacher_output.global_features[self.target_layer]
            student_global = student_output.global_features[self.target_layer]
            teacher_selected = teacher_global[:, self.student_frame_indices, :, :]

            if self.base_loss_type == 'cosine':
                cos_sim = F.cosine_similarity(student_global, teacher_selected, dim=-1)
                base_loss = (1 - cos_sim) / self.cosine_temperature
                base_loss = base_loss.mean()
                loss_details['base_cosine_loss'] = base_loss.item()
                loss_details['cos_sim_mean'] = cos_sim.mean().item()

            elif self.base_loss_type == 'mse':
                base_loss = F.mse_loss(student_global, teacher_selected)
                loss_details['base_mse_loss'] = base_loss.item()

            elif self.base_loss_type == 'global_mse':
                base_loss = F.mse_loss(student_global, teacher_selected)
                loss_details['base_global_mse_loss'] = base_loss.item()

            else:
                raise ValueError(f"Unknown base_loss_type: {self.base_loss_type}")

            total_loss = total_loss + self.base_weight * base_loss
            loss_details['base_loss'] = base_loss.item()

        # Compute attention distillation loss
        attn_loss, attn_details = self.attn_loss(teacher_output, student_output)
        total_loss = total_loss + self.attn_weight * attn_loss

        # Merge details
        for k, v in attn_details.items():
            if k != 'total_loss':
                loss_details[k] = v

        loss_details['total_loss'] = total_loss.item()

        return total_loss, loss_details

    def extra_repr(self) -> str:
        return (
            f"base_loss_type={self.base_loss_type}, "
            f"base_weight={self.base_weight}, "
            f"attn_weight={self.attn_weight}"
        )


class LocalGlobalNormCosineLoss(nn.Module):
    """
    Normalized cosine loss for both local and global features (separately).

    - Local: L2-normalize then compute cosine similarity
    - Global: L2-normalize then compute cosine similarity

    No robust loss, only cosine similarity on normalized features.
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        output_layers: List[int] = None,
        local_weight: float = 1.0,
        global_weight: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.target_layer = 39
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.temperature = temperature
        print(f"LocalGlobalNormCosineLoss: layer={self.target_layer}")
        print(f"  Local: weight={local_weight}, temperature={temperature}")
        print(f"  Global: weight={global_weight}, temperature={temperature}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Get features
        teacher_local = teacher_output.local_features[self.target_layer]
        student_local = student_output.local_features[self.target_layer]
        teacher_global = teacher_output.global_features[self.target_layer]
        student_global = student_output.global_features[self.target_layer]

        # Select teacher frames corresponding to student frames
        teacher_local_sel = teacher_local[:, self.student_frame_indices, :, :]
        teacher_global_sel = teacher_global[:, self.student_frame_indices, :, :]

        # Local: L2-normalize then cosine similarity
        teacher_local_norm = F.normalize(teacher_local_sel, dim=-1)
        student_local_norm = F.normalize(student_local, dim=-1)
        local_cos_sim = F.cosine_similarity(student_local_norm, teacher_local_norm, dim=-1)
        local_cosine = ((1 - local_cos_sim) / self.temperature).mean()

        # Global: L2-normalize then cosine similarity
        teacher_global_norm = F.normalize(teacher_global_sel, dim=-1)
        student_global_norm = F.normalize(student_global, dim=-1)
        global_cos_sim = F.cosine_similarity(student_global_norm, teacher_global_norm, dim=-1)
        global_cosine = ((1 - global_cos_sim) / self.temperature).mean()

        # Combine losses
        total = (
            self.local_weight * local_cosine +
            self.global_weight * global_cosine
        )

        return total, {
            'local_cosine_loss': local_cosine.item(),
            'local_cos_sim_mean': local_cos_sim.mean().item(),
            'global_cosine_loss': global_cosine.item(),
            'global_cos_sim_mean': global_cos_sim.mean().item(),
            'total_loss': total.item(),
        }


def create_distillation_loss(
    output_layers: List[int] = None,
    robust_alpha: float = 0.5,
    robust_scaling_c: float = 0.25,
    cosine_temperature: float = 0.1,
    feature_weights: Dict[str, float] = None,
    camera_token_weight: float = 1.0,
) -> DA3DistillationLoss:
    """
    Factory function to create DA3DistillationLoss with common configurations.

    Args:
        output_layers: Layer indices to compute loss at
        robust_alpha: Alpha for robust loss
        robust_scaling_c: Scaling for robust loss
        cosine_temperature: Temperature for cosine loss
        feature_weights: Weights for 'robust' and 'cosine' losses
        camera_token_weight: Weight for camera token loss

    Returns:
        DA3DistillationLoss instance
    """
    return DA3DistillationLoss(
        output_layers=output_layers,
        feature_loss_weights=feature_weights,
        camera_token_weight=camera_token_weight,
        robust_alpha=robust_alpha,
        robust_scaling_c=robust_scaling_c,
        cosine_temperature=cosine_temperature,
    )
