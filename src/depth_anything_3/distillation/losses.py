"""
Distillation Loss Functions for DA3 Knowledge Distillation.

This module provides loss functions for distilling knowledge from a teacher
model (8 views) to a student model (4 views).

Key classes:
- LocalTokenSoftmaxKLCosineLoss: Local token per-token channel softmax KL + cosine
- GlobalTokenSoftmaxKLCosineLoss: Global token per-token channel softmax KL + cosine
- CombinedTokenSoftmaxKLCosineLoss: Combined local+global softmax KL + cosine
- AllTokenSoftmaxKLCosineLoss: Full 3072-dim token softmax KL + cosine (no split)

Optional extension:
- AllTokenSoftmaxKLCosineLoss can also add a "Gaussian KL" term where patch tokens
  are treated as samples from a diagonal-covariance Gaussian in R^C (C=3072).
  This keeps the original softmax-KL (distribution over channels) while adding
  a second KL that matches 1st/2nd-order patch-token statistics.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from depth_anything_3.distillation.models import DistillationOutput


def _gaussian_kl_diag(
    mu_p: torch.Tensor,
    var_p: torch.Tensor,
    mu_q: torch.Tensor,
    var_q: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-dimension KL between diagonal Gaussians.

    Computes KL( N(mu_p, diag(var_p)) || N(mu_q, diag(var_q)) ) and then averages
    over the last (feature) dimension. Averaging keeps the magnitude roughly
    invariant to C (e.g., 3072) so the weight is easier to tune.

    Shapes:
      mu_*, var_*: [..., C]
    Returns:
      kl: [...] (C averaged out)
    """
    var_p = var_p.clamp_min(eps)
    var_q = var_q.clamp_min(eps)
    return 0.5 * (
        torch.log(var_q / var_p)
        + (var_p + (mu_p - mu_q) ** 2) / var_q
        - 1.0
    ).mean(dim=-1)


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
        gaussian_kl_weight: float = 0.0,
        gaussian_kl_eps: float = 1e-6,
        target_layers: List[int] = None,
        temperature: float = 1.0,
        l2_normalize: bool = False,
    ):
        super().__init__()
        self.target_layers = target_layers or [39]
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.kl_weight = kl_weight
        self.cos_weight = cos_weight
        self.gaussian_kl_weight = gaussian_kl_weight
        self.gaussian_kl_eps = gaussian_kl_eps
        self.temperature = temperature
        self.l2_normalize = l2_normalize
        print(f"AllTokenSoftmaxKLCosineLoss: layers={self.target_layers}, "
              f"kl_w={kl_weight}, cos_w={cos_weight}, "
              f"gauss_kl_w={gaussian_kl_weight}, "
              f"temp={temperature}, l2_norm={l2_normalize}")
        print("  Uses full 3072-dim token (no local/global split)")
        if self.gaussian_kl_weight > 0.0:
            print("  Adds diagonal-Gaussian KL on patch-token stats (mean/var over patches)")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_kl = 0.0
        total_cos = 0.0
        total_gauss_kl = 0.0

        for layer in self.target_layers:
            teacher_all = teacher_output.layer_features[layer]  # [B, 8, P, 3072]
            student_all = student_output.layer_features[layer]  # [B, 4, P, 3072]
            teacher_selected = teacher_all[:, self.student_frame_indices, :, :]

            # KL branch: optionally L2-normalize then apply temperature
            t_kl = teacher_selected
            s_kl = student_all
            if self.l2_normalize:
                t_kl = torch.nn.functional.normalize(t_kl, dim=-1)
                s_kl = torch.nn.functional.normalize(s_kl, dim=-1)

            teacher_sm = torch.softmax(t_kl / self.temperature, dim=-1)
            student_sm = torch.softmax(s_kl / self.temperature, dim=-1)
            kl = (teacher_sm * (torch.log(teacher_sm + 1e-8) - torch.log(student_sm + 1e-8))).sum(dim=-1).mean()
            # Scale by T^2 to keep gradient magnitude consistent (Hinton et al.)
            kl = kl * (self.temperature ** 2)

            teacher_norm = torch.nn.functional.normalize(teacher_selected, dim=-1)
            student_norm = torch.nn.functional.normalize(student_all, dim=-1)
            cos = (teacher_norm * student_norm).sum(dim=-1).mean()

            total_kl = total_kl + kl
            total_cos = total_cos + cos

            # Optional: diagonal Gaussian KL between patch-token distributions in R^C.
            # Treat patches as samples: for each (B, view), compute mean/var over P.
            if self.gaussian_kl_weight > 0.0:
                mu_t = teacher_selected.mean(dim=2)
                var_t = teacher_selected.var(dim=2, unbiased=False)
                mu_s = student_all.mean(dim=2)
                var_s = student_all.var(dim=2, unbiased=False)
                gauss_kl = _gaussian_kl_diag(mu_t, var_t, mu_s, var_s, eps=self.gaussian_kl_eps).mean()
                total_gauss_kl = total_gauss_kl + gauss_kl

        n = len(self.target_layers)
        avg_kl = total_kl / n
        avg_cos = total_cos / n
        avg_gauss_kl = total_gauss_kl / n if self.gaussian_kl_weight > 0.0 else torch.tensor(0.0, device=avg_kl.device)

        loss = (
            self.kl_weight * avg_kl
            + self.gaussian_kl_weight * avg_gauss_kl
            + self.cos_weight * (1.0 - avg_cos)
        )

        details = {
            'all_softmax_kl': avg_kl.item(),
            'all_softmax_cos': avg_cos.item(),
            'all_softmax_total': loss.item(),
            'total_loss': loss.item(),
        }
        if self.gaussian_kl_weight > 0.0:
            details['all_gaussian_kl'] = avg_gauss_kl.item()
        return loss, details


class PatchL2CosineLoss(nn.Module):
    """Per-patch L2 + cosine loss on the full 3072-dim token.

    L2 matches absolute feature values; cosine matches direction.
    Both operate on the full concatenated [local, global] token directly.
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        l2_weight: float = 1.0,
        cos_weight: float = 2.0,
        target_layers: List[int] = None,
    ):
        super().__init__()
        self.target_layers = target_layers or [39]
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.l2_weight = l2_weight
        self.cos_weight = cos_weight
        print(f"PatchL2CosineLoss: layers={self.target_layers}, "
              f"l2_w={l2_weight}, cos_w={cos_weight}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_l2 = 0.0
        total_cos = 0.0

        for layer in self.target_layers:
            teacher_all = teacher_output.layer_features[layer]  # [B, 8, P, 3072]
            student_all = student_output.layer_features[layer]  # [B, 4, P, 3072]
            teacher_selected = teacher_all[:, self.student_frame_indices, :, :].detach()

            l2 = ((teacher_selected - student_all) ** 2).mean()

            teacher_norm = torch.nn.functional.normalize(teacher_selected, dim=-1)
            student_norm = torch.nn.functional.normalize(student_all, dim=-1)
            cos = (teacher_norm * student_norm).sum(dim=-1).mean()

            total_l2 = total_l2 + l2
            total_cos = total_cos + cos

        n = len(self.target_layers)
        avg_l2 = total_l2 / n
        avg_cos = total_cos / n

        loss = self.l2_weight * avg_l2 + self.cos_weight * (1.0 - avg_cos)

        return loss, {
            'patch_l2': avg_l2.item(),
            'patch_l2_cos': avg_cos.item(),
            'patch_l2_total': loss.item(),
            'total_loss': loss.item(),
        }


class PatchHuberCosineLoss(nn.Module):
    """Per-patch Huber (SmoothL1) + cosine loss on the full 3072-dim token.

    Huber is less sensitive to outliers than L2 — quadratic for small errors,
    linear for large ones. Delta controls the transition point.
    """

    def __init__(
        self,
        student_frame_indices: List[int] = None,
        huber_weight: float = 1.0,
        cos_weight: float = 2.0,
        target_layers: List[int] = None,
        delta: float = 1.0,
    ):
        super().__init__()
        self.target_layers = target_layers or [39]
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.huber_weight = huber_weight
        self.cos_weight = cos_weight
        self.huber = nn.SmoothL1Loss(reduction='mean', beta=delta)
        print(f"PatchHuberCosineLoss: layers={self.target_layers}, "
              f"huber_w={huber_weight}, cos_w={cos_weight}, delta={delta}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_huber = 0.0
        total_cos = 0.0

        for layer in self.target_layers:
            teacher_all = teacher_output.layer_features[layer]
            student_all = student_output.layer_features[layer]
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


class DA3CrossFrameRKDAngleLoss(nn.Module):
    """
    Cross-Frame RKD Angle-Wise Loss for DA3 distillation.

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
        student_frame_indices: List[int] | None = None,
        num_teacher_views: int = 8,
        target_layer: int = 39,
        topk: int = 4,
        num_ref_samples: int = 256,
        num_shared_samples: int = 256,
        angle1_weight: float = 1.0,
        angle2_weight: float = 1.0,
        angle3_weight: float = 1.0,
        shared_chunk_size: int = 32,
        selection_mode: str = 'topk',  # 'topk', 'random', 'mixed'
        # Legacy params accepted but ignored
        num_triplets: int = 4096,
        huber_delta: float = 1.0,
    ) -> None:
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
        self.l1_loss = nn.L1Loss(reduction='none')
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

        print(f"DA3CrossFrameRKDAngleLoss: layer={target_layer}, topk={topk}, selection={selection_mode}")
        print(f"  ref_samples={num_ref_samples}, shared_samples={num_shared_samples}, shared_chunk={shared_chunk_size}")
        print(f"  angle weights: a1={angle1_weight}, a2={angle2_weight}, a3={angle3_weight}")
        print(f"  extra frames (teacher-only): {self.extra_frame_indices}")
        print(f"  shared frames: {self.shared_frame_teacher_indices}")
        print(f"  loss: L1")

    @staticmethod
    def _cos_angle(vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
        a_norm = torch.nn.functional.normalize(vec_a, dim=-1, eps=1e-8)
        b_norm = torch.nn.functional.normalize(vec_b, dim=-1, eps=1e-8)
        return (a_norm * b_norm).sum(dim=-1)

    def _compute_chunk_angles(
        self,
        ref_t_4d: torch.Tensor,
        ref_s_4d: torch.Tensor,
        shared_t_4d: torch.Tensor,
        shared_s_4d: torch.Tensor,
        sim_high_4d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute angle losses for one chunk. Called inside checkpoint."""
        # Angle 1: vertex = ref
        a1_t = self._cos_angle(shared_t_4d - ref_t_4d, sim_high_4d - ref_t_4d)
        a1_s = self._cos_angle(shared_s_4d - ref_s_4d, sim_high_4d - ref_s_4d)
        chunk_a1 = self.l1_loss(a1_s, a1_t.detach()).sum()

        # Angle 2: vertex = sim_high
        a2_t = self._cos_angle(ref_t_4d - sim_high_4d, shared_t_4d - sim_high_4d)
        a2_s = self._cos_angle(ref_s_4d - sim_high_4d, shared_s_4d - sim_high_4d)
        chunk_a2 = self.l1_loss(a2_s, a2_t.detach()).sum()

        # Angle 3: vertex = shared
        a3_t = self._cos_angle(ref_t_4d - shared_t_4d, sim_high_4d - shared_t_4d)
        a3_s = self._cos_angle(ref_s_4d - shared_s_4d, sim_high_4d - shared_s_4d)
        chunk_a3 = self.l1_loss(a3_s, a3_t.detach()).sum()

        return chunk_a1, chunk_a2, chunk_a3

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_feats = teacher_output.layer_features[self.target_layer]  # [B, S_t, P, D]
        student_feats = student_output.layer_features[self.target_layer]  # [B, S_s, P, D]

        B, S_t, P, D = teacher_feats.shape

        # --- 1. Subsample positions upfront ---
        num_ref = min(self.num_ref_samples, P)
        num_shared = min(self.num_shared_samples, P)
        ref_perm = torch.randperm(P, device=teacher_feats.device)[:num_ref]
        shared_perm = torch.randperm(P, device=teacher_feats.device)[:num_shared]

        # --- 2. Extract only the slices we need ---
        ref_t_sampled = teacher_feats[:, self.ref_frame_teacher_idx, ref_perm, :].detach()  # [B, num_ref, D]
        ref_s_idx = self.teacher_to_student[self.ref_frame_teacher_idx]
        ref_s_sampled = student_feats[:, ref_s_idx, ref_perm, :]  # [B, num_ref, D]

        # --- 3. Select patches across extra frames (no grad) ---
        with torch.no_grad():
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
        # Use gradient checkpointing per chunk to avoid accumulating the entire
        # autograd graph across all chunks. Each chunk's forward is recomputed
        # during backward, keeping peak memory proportional to one chunk.
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

                    chunk_a1, chunk_a2, chunk_a3 = torch.utils.checkpoint.checkpoint(
                        self._compute_chunk_angles,
                        ref_t_4d, ref_s_4d, shared_t_4d, shared_s_4d, sim_high_4d,
                        use_reentrant=False,
                    )
                    sum_angle1 = sum_angle1 + chunk_a1
                    sum_angle2 = sum_angle2 + chunk_a2
                    sum_angle3 = sum_angle3 + chunk_a3
                    del chunk_a1, chunk_a2, chunk_a3

                    total_elements += n_elem

        # Mean over all elements
        total_angle1_loss = sum_angle1 / total_elements
        total_angle2_loss = sum_angle2 / total_elements
        total_angle3_loss = sum_angle3 / total_elements

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


class DA3CrossFrameRKDDistanceLoss(nn.Module):
    """
    Cross-Frame RKD Distance-Wise Loss for DA3 distillation.

    Complementary to DA3CrossFrameRKDAngleLoss. Instead of distilling the
    angles of the triplet (ref, shared, sim_high), this loss distills the
    pairwise distances of the three edges:
      d1 = ||ref - shared||
      d2 = ||ref - sim_high||
      d3 = ||shared - sim_high||

    Combined with the angle loss, this captures the full triangle geometry
    (both shape and scale).

    Uses the same subsampling, top-K search, and chunked computation as the
    angle loss for memory efficiency.
    """

    def __init__(
        self,
        student_frame_indices: List[int] | None = None,
        num_teacher_views: int = 8,
        target_layer: int = 39,
        topk: int = 4,
        num_ref_samples: int = 256,
        num_shared_samples: int = 256,
        d1_weight: float = 1.0,
        d2_weight: float = 1.0,
        d3_weight: float = 1.0,
        shared_chunk_size: int = 64,
        distance_chunk_size: int = 16,
        distance_type: str = 'l2',
        normalize_distance: bool = True,
        temperature: float = 1.0,
        distance_mode: str = 'kl',  # 'kl' = softmax KL on diffs, 'huber' = direct Huber on diffs
        huber_beta: float = 0.5,    # beta for Huber loss when distance_mode='huber'
        selection_mode: str = 'topk',  # 'topk', 'random', 'mixed'
        # Legacy params accepted but ignored (match angle loss interface)
        num_triplets: int = 4096,
        huber_delta: float = 1.0,
    ) -> None:
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

        print(f"DA3CrossFrameRKDDistanceLoss: layer={target_layer}, topk={topk}, temp={temperature}, mode={distance_mode}, selection={selection_mode}")
        print(f"  ref_samples={num_ref_samples}, shared_samples={num_shared_samples}, shared_chunk={shared_chunk_size}, distance_chunk={distance_chunk_size}")
        print(f"  distance weights: d1={d1_weight}, d2={d2_weight}, d3={d3_weight}")
        print(f"  distance_type={distance_type}, normalize={normalize_distance}")
        if distance_mode == 'huber':
            print(f"  huber_beta={huber_beta}")
        print(f"  extra frames (teacher-only): {self.extra_frame_indices}")
        print(f"  shared frames: {self.shared_frame_teacher_indices}")

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_feats = teacher_output.layer_features[self.target_layer]  # [B, S_t, P, D]
        student_feats = student_output.layer_features[self.target_layer]  # [B, S_s, P, D]

        B, S_t, P, D = teacher_feats.shape

        # --- 1. Subsample positions upfront ---
        num_ref = min(self.num_ref_samples, P)
        num_shared = min(self.num_shared_samples, P)
        ref_perm = torch.randperm(P, device=teacher_feats.device)[:num_ref]
        shared_perm = torch.randperm(P, device=teacher_feats.device)[:num_shared]

        # --- 2. Extract only the slices we need ---
        ref_t_sampled = teacher_feats[:, self.ref_frame_teacher_idx, ref_perm, :].detach()  # [B, num_ref, D]
        ref_s_idx = self.teacher_to_student[self.ref_frame_teacher_idx]
        ref_s_sampled = student_feats[:, ref_s_idx, ref_perm, :]  # [B, num_ref, D]

        # --- 3. Select patches across extra frames (no grad) ---
        with torch.no_grad():
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

        # --- 4. Compute per-dim difference profiles ---
        N = min(num_ref, num_shared)
        cs = self.distance_chunk_size
        T = self.temperature
        huber_outer = nn.SmoothL1Loss(reduction='none', beta=0.5)
        huber_direct = nn.SmoothL1Loss(reduction='none', beta=self.huber_beta)

        sum_d1 = torch.tensor(0.0, device=teacher_feats.device)
        sum_d2 = torch.tensor(0.0, device=teacher_feats.device)
        sum_d3 = torch.tensor(0.0, device=teacher_feats.device)
        total_d1_elements = 0
        total_d2_elements = 0
        total_d3_elements = 0

        ref_t_paired = ref_t_sampled[:, :N, :]   # [B, N, D]
        ref_s_paired = ref_s_sampled[:, :N, :]
        sim_high_paired = sim_high_t[:, :N, :, :]  # [B, N, K, D]

        for shared_teacher_idx in self.shared_frame_teacher_indices:
            shared_student_idx = self.teacher_to_student[shared_teacher_idx]

            shared_t = teacher_feats[:, shared_teacher_idx, shared_perm[:N], :].detach()  # [B, N, D]
            shared_s = student_feats[:, shared_student_idx, shared_perm[:N], :]

            for c0 in range(0, N, cs):
                c1 = min(c0 + cs, N)

                rt = ref_t_paired[:, c0:c1, :].detach()  # [B, cs, D]
                rs = ref_s_paired[:, c0:c1, :]
                st = shared_t[:, c0:c1, :]                # [B, cs, D]
                ss = shared_s[:, c0:c1, :]
                sh = sim_high_paired[:, c0:c1, :, :]      # [B, cs, K, D]

                # d1: ref - shared (paired) -> [B, cs, D]
                diff_d1_t = rt - st
                diff_d1_s = rs - ss

                if self.distance_mode == 'huber':
                    # Direct Huber on difference vectors
                    loss_d1_chunk = huber_direct(diff_d1_s, diff_d1_t).mean(dim=-1)  # [B, cs]
                    sum_d1 = sum_d1 + loss_d1_chunk.sum()
                    total_d1_elements += loss_d1_chunk.numel()
                    del diff_d1_t, diff_d1_s, loss_d1_chunk
                else:
                    log_p = torch.log_softmax(diff_d1_t / T, dim=-1)
                    log_q = torch.log_softmax(diff_d1_s / T, dim=-1)
                    p = log_p.exp()
                    kl_d1 = (p * (log_p - log_q)).sum(dim=-1)  # [B, cs]
                    sum_d1 = sum_d1 + huber_outer(kl_d1, torch.zeros_like(kl_d1)).sum()
                    total_d1_elements += kl_d1.numel()
                    del diff_d1_t, diff_d1_s, log_p, log_q, p, kl_d1

                # d2: ref - sim_high (paired) -> [B, cs, K, D]
                diff_d2_t = rt.unsqueeze(2) - sh  # [B, cs, K, D]
                diff_d2_s = rs.unsqueeze(2) - sh

                if self.distance_mode == 'huber':
                    loss_d2_chunk = huber_direct(diff_d2_s, diff_d2_t).mean(dim=-1)  # [B, cs, K]
                    sum_d2 = sum_d2 + loss_d2_chunk.sum()
                    total_d2_elements += loss_d2_chunk.numel()
                    del diff_d2_t, diff_d2_s, loss_d2_chunk
                else:
                    log_p = torch.log_softmax(diff_d2_t / T, dim=-1)
                    log_q = torch.log_softmax(diff_d2_s / T, dim=-1)
                    p = log_p.exp()
                    kl_d2 = (p * (log_p - log_q)).sum(dim=-1)  # [B, cs, K]
                    sum_d2 = sum_d2 + huber_outer(kl_d2, torch.zeros_like(kl_d2)).sum()
                    total_d2_elements += kl_d2.numel()
                    del diff_d2_t, diff_d2_s, log_p, log_q, p, kl_d2

                # d3: shared - sim_high (paired) -> [B, cs, K, D]
                diff_d3_t = st.unsqueeze(2) - sh  # [B, cs, K, D]
                diff_d3_s = ss.unsqueeze(2) - sh

                if self.distance_mode == 'huber':
                    loss_d3_chunk = huber_direct(diff_d3_s, diff_d3_t).mean(dim=-1)  # [B, cs, K]
                    sum_d3 = sum_d3 + loss_d3_chunk.sum()
                    total_d3_elements += loss_d3_chunk.numel()
                    del diff_d3_t, diff_d3_s, loss_d3_chunk
                else:
                    log_p = torch.log_softmax(diff_d3_t / T, dim=-1)
                    log_q = torch.log_softmax(diff_d3_s / T, dim=-1)
                    p = log_p.exp()
                    kl_d3 = (p * (log_p - log_q)).sum(dim=-1)  # [B, cs, K]
                    sum_d3 = sum_d3 + huber_outer(kl_d3, torch.zeros_like(kl_d3)).sum()
                    total_d3_elements += kl_d3.numel()
                    del diff_d3_t, diff_d3_s, log_p, log_q, p, kl_d3

        # --- 5. Mean KL losses with Huber dampening ---
        loss_d1 = sum_d1 / max(total_d1_elements, 1)
        loss_d2 = sum_d2 / max(total_d2_elements, 1)
        loss_d3 = sum_d3 / max(total_d3_elements, 1)

        loss = self.d1_weight * loss_d1 + self.d2_weight * loss_d2 + self.d3_weight * loss_d3

        return loss, {
            'rkd_dist_d1_loss': loss_d1.item(),
            'rkd_dist_d2_loss': loss_d2.item(),
            'rkd_dist_d3_loss': loss_d3.item(),
            'rkd_dist_total_loss': loss.item(),
        }


class DA3CameraTokenRKDLoss(nn.Module):
    """
    Camera-token RKD angle-wise loss for DA3 distillation.

    Uses only the camera tokens (one per frame) and exhaustively computes
    all triplets of (ref_cam, shared_cam, extra_cam). With 1 ref, 3 shared,
    and 4 extra frames this gives 3*4=12 triplets per batch — every one
    captures a real viewpoint relationship.
    """

    def __init__(
        self,
        student_frame_indices: List[int] | None = None,
        num_teacher_views: int = 8,
        target_layer: int = 39,
        angle1_weight: float = 1.0,
        angle2_weight: float = 1.0,
        angle3_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.num_teacher_views = num_teacher_views
        self.target_layer = target_layer
        self.angle1_weight = angle1_weight
        self.angle2_weight = angle2_weight
        self.angle3_weight = angle3_weight

        all_teacher = set(range(num_teacher_views))
        student_set = set(self.student_frame_indices)
        self.extra_frame_indices = sorted(all_teacher - student_set)

        self.ref_frame_teacher_idx = self.student_frame_indices[0]
        self.shared_frame_teacher_indices = self.student_frame_indices[1:]

        self.teacher_to_student = {t_idx: s_idx for s_idx, t_idx in enumerate(self.student_frame_indices)}

        num_triplets = len(self.shared_frame_teacher_indices) * len(self.extra_frame_indices)
        print(f"DA3CameraTokenRKDLoss: layer={target_layer}, {num_triplets} exhaustive triplets")
        print(f"  angle weights: a1={angle1_weight}, a2={angle2_weight}, a3={angle3_weight}")
        print(f"  ref frame: {self.ref_frame_teacher_idx}")
        print(f"  shared frames: {self.shared_frame_teacher_indices}")
        print(f"  extra frames: {self.extra_frame_indices}")

    @staticmethod
    def _cos_angle(vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
        a_norm = torch.nn.functional.normalize(vec_a, dim=-1, eps=1e-8)
        b_norm = torch.nn.functional.normalize(vec_b, dim=-1, eps=1e-8)
        return (a_norm * b_norm).sum(dim=-1)

    def forward(
        self,
        teacher_output: DistillationOutput,
        student_output: DistillationOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Camera tokens: [B, S, D] (1536-dim, global half)
        teacher_cam = teacher_output.camera_tokens[self.target_layer]  # [B, 8, D]
        student_cam = student_output.camera_tokens[self.target_layer]  # [B, 4, D]

        device = teacher_cam.device
        ref_t_idx = self.ref_frame_teacher_idx
        ref_s_idx = self.teacher_to_student[ref_t_idx]

        # Reference camera token: [B, D]
        ref_t = teacher_cam[:, ref_t_idx, :].detach()
        ref_s = student_cam[:, ref_s_idx, :]

        sum_a1 = torch.tensor(0.0, device=device)
        sum_a2 = torch.tensor(0.0, device=device)
        sum_a3 = torch.tensor(0.0, device=device)
        count = 0

        for shared_t_idx in self.shared_frame_teacher_indices:
            shared_s_idx = self.teacher_to_student[shared_t_idx]
            sh_t = teacher_cam[:, shared_t_idx, :].detach()  # [B, D]
            sh_s = student_cam[:, shared_s_idx, :]

            for extra_idx in self.extra_frame_indices:
                ex_t = teacher_cam[:, extra_idx, :].detach()  # [B, D]

                # Angle 1: vertex = ref
                a1_t = self._cos_angle(sh_t - ref_t, ex_t - ref_t)
                a1_s = self._cos_angle(sh_s - ref_s, ex_t - ref_s)
                sum_a1 = sum_a1 + (a1_s - a1_t.detach()).abs().mean()

                # Angle 2: vertex = extra
                a2_t = self._cos_angle(ref_t - ex_t, sh_t - ex_t)
                a2_s = self._cos_angle(ref_s - ex_t, sh_s - ex_t)
                sum_a2 = sum_a2 + (a2_s - a2_t.detach()).abs().mean()

                # Angle 3: vertex = shared
                a3_t = self._cos_angle(ref_t - sh_t, ex_t - sh_t)
                a3_s = self._cos_angle(ref_s - sh_s, ex_t - sh_s)
                sum_a3 = sum_a3 + (a3_s - a3_t.detach()).abs().mean()

                count += 1

        loss_a1 = sum_a1 / count
        loss_a2 = sum_a2 / count
        loss_a3 = sum_a3 / count

        loss = (
            self.angle1_weight * loss_a1
            + self.angle2_weight * loss_a2
            + self.angle3_weight * loss_a3
        )

        return loss, {
            "cam_rkd_angle1_loss": loss_a1.item(),
            "cam_rkd_angle2_loss": loss_a2.item(),
            "cam_rkd_angle3_loss": loss_a3.item(),
            "cam_rkd_total_loss": loss.item(),
        }
