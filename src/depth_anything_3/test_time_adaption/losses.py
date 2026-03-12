"""Active DA3 Free-Geometry losses used by the current training scripts."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from depth_anything_3.test_time_adaption.models import FreeGeometryOutput


class PatchHuberCosineLoss(nn.Module):
    """Per-patch Huber + cosine loss on the full token."""

    def __init__(
        self,
        student_frame_indices: List[int] | None = None,
        huber_weight: float = 1.0,
        cos_weight: float = 2.0,
        target_layers: List[int] | None = None,
        delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.target_layers = target_layers or [39]
        self.student_frame_indices = student_frame_indices or [0, 2, 4, 6]
        self.huber_weight = huber_weight
        self.cos_weight = cos_weight
        self.huber = nn.SmoothL1Loss(reduction="mean", beta=delta)
        print(
            f"PatchHuberCosineLoss: layers={self.target_layers}, "
            f"huber_w={huber_weight}, cos_w={cos_weight}, delta={delta}"
        )

    def forward(
        self,
        teacher_output: FreeGeometryOutput,
        student_output: FreeGeometryOutput,
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

        n_layers = len(self.target_layers)
        avg_huber = total_huber / n_layers
        avg_cos = total_cos / n_layers
        loss = self.huber_weight * avg_huber + self.cos_weight * (1.0 - avg_cos)

        return loss, {
            "patch_huber": avg_huber.item(),
            "patch_huber_cos": avg_cos.item(),
            "patch_huber_total": loss.item(),
            "total_loss": loss.item(),
        }


class DA3CrossFrameCFAngleLoss(nn.Module):
    """Cross-frame CF angle loss for DA3."""

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
        selection_mode: str = "topk",
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
        self.l1_loss = nn.L1Loss(reduction="none")
        self.shared_chunk_size = shared_chunk_size
        self.selection_mode = selection_mode

        all_teacher_indices = set(range(num_teacher_views))
        student_set = set(self.student_frame_indices)
        self.extra_frame_indices = sorted(all_teacher_indices - student_set)
        self.ref_frame_teacher_idx = self.student_frame_indices[0]
        self.shared_frame_teacher_indices = self.student_frame_indices[1:]
        self.teacher_to_student = {
            t_idx: s_idx for s_idx, t_idx in enumerate(self.student_frame_indices)
        }

        print(f"DA3CrossFrameCFAngleLoss: layer={target_layer}, topk={topk}, selection={selection_mode}")
        print(
            f"  ref_samples={num_ref_samples}, shared_samples={num_shared_samples}, "
            f"shared_chunk={shared_chunk_size}"
        )
        print(f"  angle weights: a1={angle1_weight}, a2={angle2_weight}, a3={angle3_weight}")
        print(f"  extra frames (teacher-only): {self.extra_frame_indices}")
        print(f"  shared frames: {self.shared_frame_teacher_indices}")
        print("  loss: L1")

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
        angle1_t = self._cos_angle(shared_t_4d - ref_t_4d, sim_high_4d - ref_t_4d)
        angle1_s = self._cos_angle(shared_s_4d - ref_s_4d, sim_high_4d - ref_s_4d)
        chunk_a1 = self.l1_loss(angle1_s, angle1_t.detach()).sum()

        angle2_t = self._cos_angle(ref_t_4d - sim_high_4d, shared_t_4d - sim_high_4d)
        angle2_s = self._cos_angle(ref_s_4d - sim_high_4d, shared_s_4d - sim_high_4d)
        chunk_a2 = self.l1_loss(angle2_s, angle2_t.detach()).sum()

        angle3_t = self._cos_angle(ref_t_4d - shared_t_4d, sim_high_4d - shared_t_4d)
        angle3_s = self._cos_angle(ref_s_4d - shared_s_4d, sim_high_4d - shared_s_4d)
        chunk_a3 = self.l1_loss(angle3_s, angle3_t.detach()).sum()

        return chunk_a1, chunk_a2, chunk_a3

    def forward(
        self,
        teacher_output: FreeGeometryOutput,
        student_output: FreeGeometryOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_feats = teacher_output.layer_features[self.target_layer]
        student_feats = student_output.layer_features[self.target_layer]

        batch_size, _, num_patches, dim = teacher_feats.shape
        num_ref = min(self.num_ref_samples, num_patches)
        num_shared = min(self.num_shared_samples, num_patches)
        ref_perm = torch.randperm(num_patches, device=teacher_feats.device)[:num_ref]
        shared_perm = torch.randperm(num_patches, device=teacher_feats.device)[:num_shared]

        ref_t_sampled = teacher_feats[:, self.ref_frame_teacher_idx, ref_perm, :].detach()
        ref_s_idx = self.teacher_to_student[self.ref_frame_teacher_idx]
        ref_s_sampled = student_feats[:, ref_s_idx, ref_perm, :]

        with torch.no_grad():
            extra_t = torch.cat(
                [teacher_feats[:, frame_idx, :, :] for frame_idx in self.extra_frame_indices],
                dim=1,
            )

            if self.selection_mode == "random":
                ep = extra_t.shape[1]
                rand_indices = torch.randint(
                    0,
                    ep,
                    (batch_size, num_ref, self.topk),
                    device=teacher_feats.device,
                )
                sim_high_t = torch.zeros(
                    batch_size,
                    num_ref,
                    self.topk,
                    dim,
                    device=teacher_feats.device,
                    dtype=teacher_feats.dtype,
                )
                for batch_idx in range(batch_size):
                    flat_idx = rand_indices[batch_idx].reshape(-1)
                    sim_high_t[batch_idx] = extra_t[batch_idx, flat_idx, :].reshape(
                        num_ref, self.topk, dim
                    )
            else:
                ref_t_norm = torch.nn.functional.normalize(ref_t_sampled, dim=-1)
                extra_t_norm = torch.nn.functional.normalize(extra_t, dim=-1)
                sim_matrix = torch.bmm(ref_t_norm, extra_t_norm.transpose(1, 2))

                if self.selection_mode == "mixed":
                    k_top = self.topk // 2
                    k_bot = self.topk - k_top
                    _, topk_indices = sim_matrix.topk(k_top, dim=-1, largest=True)
                    _, botk_indices = sim_matrix.topk(k_bot, dim=-1, largest=False)
                    gather_indices = torch.cat([topk_indices, botk_indices], dim=-1)
                else:
                    _, gather_indices = sim_matrix.topk(self.topk, dim=-1)

                sim_high_t = torch.zeros(
                    batch_size,
                    num_ref,
                    self.topk,
                    dim,
                    device=teacher_feats.device,
                    dtype=teacher_feats.dtype,
                )
                for batch_idx in range(batch_size):
                    flat_idx = gather_indices[batch_idx].reshape(-1)
                    sim_high_t[batch_idx] = extra_t[batch_idx, flat_idx, :].reshape(
                        num_ref, self.topk, dim
                    )

        sim_high_t = sim_high_t.detach()

        sum_angle1 = torch.tensor(0.0, device=teacher_feats.device)
        sum_angle2 = torch.tensor(0.0, device=teacher_feats.device)
        sum_angle3 = torch.tensor(0.0, device=teacher_feats.device)
        total_elements = 0
        chunk_size = self.shared_chunk_size

        for shared_teacher_idx in self.shared_frame_teacher_indices:
            shared_student_idx = self.teacher_to_student[shared_teacher_idx]
            shared_t_full = teacher_feats[:, shared_teacher_idx, shared_perm, :].detach()
            shared_s_full = student_feats[:, shared_student_idx, shared_perm, :]

            for ref_start in range(0, num_ref, chunk_size):
                ref_end = min(ref_start + chunk_size, num_ref)
                ref_count = ref_end - ref_start

                ref_t_4d = ref_t_sampled[:, ref_start:ref_end, :].detach().unsqueeze(2).unsqueeze(3)
                ref_s_4d = ref_s_sampled[:, ref_start:ref_end, :].unsqueeze(2).unsqueeze(3)
                sim_high_4d = sim_high_t[:, ref_start:ref_end, :, :].unsqueeze(2)

                for shared_start in range(0, num_shared, chunk_size):
                    shared_end = min(shared_start + chunk_size, num_shared)
                    shared_count = shared_end - shared_start

                    shared_t_4d = shared_t_full[:, shared_start:shared_end, :].unsqueeze(1).unsqueeze(3)
                    shared_s_4d = shared_s_full[:, shared_start:shared_end, :].unsqueeze(1).unsqueeze(3)

                    chunk_a1, chunk_a2, chunk_a3 = torch.utils.checkpoint.checkpoint(
                        self._compute_chunk_angles,
                        ref_t_4d,
                        ref_s_4d,
                        shared_t_4d,
                        shared_s_4d,
                        sim_high_4d,
                        use_reentrant=False,
                    )
                    sum_angle1 = sum_angle1 + chunk_a1
                    sum_angle2 = sum_angle2 + chunk_a2
                    sum_angle3 = sum_angle3 + chunk_a3

                    total_elements += batch_size * ref_count * shared_count * self.topk

        total_angle1_loss = sum_angle1 / total_elements
        total_angle2_loss = sum_angle2 / total_elements
        total_angle3_loss = sum_angle3 / total_elements
        loss = (
            self.angle1_weight * total_angle1_loss
            + self.angle2_weight * total_angle2_loss
            + self.angle3_weight * total_angle3_loss
        )

        return loss, {
            "cf_angle1_loss": total_angle1_loss.item(),
            "cf_angle2_loss": total_angle2_loss.item(),
            "cf_angle3_loss": total_angle3_loss.item(),
            "cf_total_loss": loss.item(),
        }


class DA3CrossFrameCFDistanceLoss(nn.Module):
    """Cross-frame CF distance loss for DA3."""

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
        distance_type: str = "l2",
        normalize_distance: bool = True,
        temperature: float = 1.0,
        distance_mode: str = "kl",
        huber_beta: float = 0.5,
        selection_mode: str = "topk",
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

        all_teacher_indices = set(range(num_teacher_views))
        student_set = set(self.student_frame_indices)
        self.extra_frame_indices = sorted(all_teacher_indices - student_set)
        self.ref_frame_teacher_idx = self.student_frame_indices[0]
        self.shared_frame_teacher_indices = self.student_frame_indices[1:]
        self.teacher_to_student = {
            t_idx: s_idx for s_idx, t_idx in enumerate(self.student_frame_indices)
        }

        print(
            f"DA3CrossFrameCFDistanceLoss: layer={target_layer}, topk={topk}, "
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
        teacher_output: FreeGeometryOutput,
        student_output: FreeGeometryOutput,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher_feats = teacher_output.layer_features[self.target_layer]
        student_feats = student_output.layer_features[self.target_layer]

        batch_size, _, num_patches, dim = teacher_feats.shape
        num_ref = min(self.num_ref_samples, num_patches)
        num_shared = min(self.num_shared_samples, num_patches)
        ref_perm = torch.randperm(num_patches, device=teacher_feats.device)[:num_ref]
        shared_perm = torch.randperm(num_patches, device=teacher_feats.device)[:num_shared]

        ref_t_sampled = teacher_feats[:, self.ref_frame_teacher_idx, ref_perm, :].detach()
        ref_s_idx = self.teacher_to_student[self.ref_frame_teacher_idx]
        ref_s_sampled = student_feats[:, ref_s_idx, ref_perm, :]

        with torch.no_grad():
            extra_t = torch.cat(
                [teacher_feats[:, frame_idx, :, :] for frame_idx in self.extra_frame_indices],
                dim=1,
            )

            if self.selection_mode == "random":
                ep = extra_t.shape[1]
                rand_indices = torch.randint(
                    0,
                    ep,
                    (batch_size, num_ref, self.topk),
                    device=teacher_feats.device,
                )
                sim_high_t = torch.zeros(
                    batch_size,
                    num_ref,
                    self.topk,
                    dim,
                    device=teacher_feats.device,
                    dtype=teacher_feats.dtype,
                )
                for batch_idx in range(batch_size):
                    flat_idx = rand_indices[batch_idx].reshape(-1)
                    sim_high_t[batch_idx] = extra_t[batch_idx, flat_idx, :].reshape(
                        num_ref, self.topk, dim
                    )
            else:
                ref_t_norm = torch.nn.functional.normalize(ref_t_sampled, dim=-1)
                extra_t_norm = torch.nn.functional.normalize(extra_t, dim=-1)
                sim_matrix = torch.bmm(ref_t_norm, extra_t_norm.transpose(1, 2))

                if self.selection_mode == "mixed":
                    k_top = self.topk // 2
                    k_bot = self.topk - k_top
                    _, topk_indices = sim_matrix.topk(k_top, dim=-1, largest=True)
                    _, botk_indices = sim_matrix.topk(k_bot, dim=-1, largest=False)
                    gather_indices = torch.cat([topk_indices, botk_indices], dim=-1)
                else:
                    _, gather_indices = sim_matrix.topk(self.topk, dim=-1)

                sim_high_t = torch.zeros(
                    batch_size,
                    num_ref,
                    self.topk,
                    dim,
                    device=teacher_feats.device,
                    dtype=teacher_feats.dtype,
                )
                for batch_idx in range(batch_size):
                    flat_idx = gather_indices[batch_idx].reshape(-1)
                    sim_high_t[batch_idx] = extra_t[batch_idx, flat_idx, :].reshape(
                        num_ref, self.topk, dim
                    )

        sim_high_t = sim_high_t.detach()

        paired_count = min(num_ref, num_shared)
        chunk_size = self.distance_chunk_size
        huber_outer = nn.SmoothL1Loss(reduction="none", beta=0.5)
        huber_direct = nn.SmoothL1Loss(reduction="none", beta=self.huber_beta)

        sum_d1 = torch.tensor(0.0, device=teacher_feats.device)
        sum_d2 = torch.tensor(0.0, device=teacher_feats.device)
        sum_d3 = torch.tensor(0.0, device=teacher_feats.device)
        total_d1_elements = 0
        total_d2_elements = 0
        total_d3_elements = 0

        ref_t_paired = ref_t_sampled[:, :paired_count, :]
        ref_s_paired = ref_s_sampled[:, :paired_count, :]
        sim_high_paired = sim_high_t[:, :paired_count, :, :]

        for shared_teacher_idx in self.shared_frame_teacher_indices:
            shared_student_idx = self.teacher_to_student[shared_teacher_idx]
            shared_t = teacher_feats[:, shared_teacher_idx, shared_perm[:paired_count], :].detach()
            shared_s = student_feats[:, shared_student_idx, shared_perm[:paired_count], :]

            for start in range(0, paired_count, chunk_size):
                end = min(start + chunk_size, paired_count)
                ref_t = ref_t_paired[:, start:end, :].detach()
                ref_s = ref_s_paired[:, start:end, :]
                shared_t_chunk = shared_t[:, start:end, :]
                shared_s_chunk = shared_s[:, start:end, :]
                sim_high = sim_high_paired[:, start:end, :, :]

                diff_d1_t = ref_t - shared_t_chunk
                diff_d1_s = ref_s - shared_s_chunk
                diff_d2_t = ref_t.unsqueeze(2) - sim_high
                diff_d2_s = ref_s.unsqueeze(2) - sim_high
                diff_d3_t = shared_t_chunk.unsqueeze(2) - sim_high
                diff_d3_s = shared_s_chunk.unsqueeze(2) - sim_high

                if self.distance_mode == "huber":
                    loss_d1_chunk = huber_direct(diff_d1_s, diff_d1_t).mean(dim=-1)
                    loss_d2_chunk = huber_direct(diff_d2_s, diff_d2_t).mean(dim=-1)
                    loss_d3_chunk = huber_direct(diff_d3_s, diff_d3_t).mean(dim=-1)
                    sum_d1 = sum_d1 + loss_d1_chunk.sum()
                    sum_d2 = sum_d2 + loss_d2_chunk.sum()
                    sum_d3 = sum_d3 + loss_d3_chunk.sum()
                    total_d1_elements += loss_d1_chunk.numel()
                    total_d2_elements += loss_d2_chunk.numel()
                    total_d3_elements += loss_d3_chunk.numel()
                else:
                    log_p_d1 = torch.log_softmax(diff_d1_t / self.temperature, dim=-1)
                    log_q_d1 = torch.log_softmax(diff_d1_s / self.temperature, dim=-1)
                    kl_d1 = (log_p_d1.exp() * (log_p_d1 - log_q_d1)).sum(dim=-1)
                    sum_d1 = sum_d1 + huber_outer(kl_d1, torch.zeros_like(kl_d1)).sum()
                    total_d1_elements += kl_d1.numel()

                    log_p_d2 = torch.log_softmax(diff_d2_t / self.temperature, dim=-1)
                    log_q_d2 = torch.log_softmax(diff_d2_s / self.temperature, dim=-1)
                    kl_d2 = (log_p_d2.exp() * (log_p_d2 - log_q_d2)).sum(dim=-1)
                    sum_d2 = sum_d2 + huber_outer(kl_d2, torch.zeros_like(kl_d2)).sum()
                    total_d2_elements += kl_d2.numel()

                    log_p_d3 = torch.log_softmax(diff_d3_t / self.temperature, dim=-1)
                    log_q_d3 = torch.log_softmax(diff_d3_s / self.temperature, dim=-1)
                    kl_d3 = (log_p_d3.exp() * (log_p_d3 - log_q_d3)).sum(dim=-1)
                    sum_d3 = sum_d3 + huber_outer(kl_d3, torch.zeros_like(kl_d3)).sum()
                    total_d3_elements += kl_d3.numel()

        loss_d1 = sum_d1 / max(total_d1_elements, 1)
        loss_d2 = sum_d2 / max(total_d2_elements, 1)
        loss_d3 = sum_d3 / max(total_d3_elements, 1)
        loss = self.d1_weight * loss_d1 + self.d2_weight * loss_d2 + self.d3_weight * loss_d3

        return loss, {
            "cf_dist_d1_loss": loss_d1.item(),
            "cf_dist_d2_loss": loss_d2.item(),
            "cf_dist_d3_loss": loss_d3.item(),
            "cf_dist_total_loss": loss.item(),
        }
