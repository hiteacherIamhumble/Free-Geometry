"""Output-level distillation losses for DA3.

This mirrors the structure of VGGT's `MultitaskLoss` (see `src/vggt/vggt/training/loss.py`),
but is adapted to DepthAnything3 (DA3) outputs.

DA3 does not expose VGGT-style point heads (world_points) and may not always
expose confidence maps. We therefore implement:

- Camera pose loss: compares student vs teacher extrinsics/intrinsics via a
  compact pose encoding (T, quat, FoV).
- Depth loss: regression + optional confidence-weighted term + optional gradient
  loss.

All functions are written to be robust to missing keys.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from depth_anything_3.model.utils.transform import extri_intri_to_pose_encoding


def _as_depth_5d(x: torch.Tensor) -> torch.Tensor:
    """Ensure depth tensor is (B, S, H, W, 1)."""
    if x.ndim == 4:
        return x[..., None]
    if x.ndim == 5:
        return x
    raise ValueError(f"Expected depth with 4 or 5 dims, got shape {tuple(x.shape)}")


def filter_by_quantile(
    loss_tensor: torch.Tensor,
    valid_range: float,
    *,
    min_elements: int = 1000,
    hard_max: float = 100.0,
) -> torch.Tensor:
    """Keep only values below the given quantile threshold (and clamp extremes)."""
    if loss_tensor.numel() <= min_elements:
        return loss_tensor
    if not (0.0 < float(valid_range) < 1.0):
        return loss_tensor
    loss_tensor = loss_tensor.clamp(max=hard_max)
    thresh = torch.quantile(loss_tensor.detach(), float(valid_range))
    thresh = torch.clamp(thresh, max=hard_max)
    mask = loss_tensor < thresh
    if mask.sum() > min_elements:
        return loss_tensor[mask]
    return loss_tensor


def gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    conf: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    alpha: float = 0.2,
) -> torch.Tensor:
    """L1 gradient loss on residuals, optionally confidence-weighted.

    Args:
        prediction/target: (B, H, W, C)
        mask: (B, H, W) bool
        conf: (B, H, W) optional
    """
    mask_c = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask_c, (1, 2, 3))
    diff = (prediction - target) * mask_c

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = mask_c[:, :, 1:] * mask_c[:, :, :-1]
    grad_x = grad_x * mask_x

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = mask_c[:, 1:, :] * mask_c[:, :-1, :]
    grad_y = grad_y * mask_y

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    if conf is not None:
        conf_c = conf[..., None].expand_as(mask_c)
        conf_x = conf_c[:, :, 1:].clamp(min=1e-6)
        conf_y = conf_c[:, 1:, :].clamp(min=1e-6)
        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    grad_sum = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    denom = torch.sum(M)
    if denom == 0:
        return (0.0 * prediction).mean()
    return torch.sum(grad_sum) / denom


def gradient_loss_multi_scale(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    scales: int = 4,
    conf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-scale wrapper around `gradient_loss`."""
    total = 0.0
    for s in range(scales):
        step = 2**s
        total = total + gradient_loss(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None,
        )
    return total / float(scales)


def regression_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    pred_conf: Optional[torch.Tensor] = None,
    gradient_loss_fn: Optional[str] = None,
    gamma: float = 1.0,
    alpha: float = 0.2,
    valid_range: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Depth regression loss with optional confidence + optional gradient loss."""
    pred = _as_depth_5d(pred_depth)
    gt = _as_depth_5d(gt_depth)

    # L1 error per-pixel
    err = torch.abs(gt[mask] - pred[mask]).squeeze(-1)
    if err.numel() > 0 and valid_range > 0:
        err = filter_by_quantile(err, valid_range)
    loss_reg = err.mean() if err.numel() > 0 else (0.0 * pred).mean()

    # Confidence-weighted term (optional)
    if pred_conf is not None:
        conf = pred_conf[mask].clamp(min=1e-6)
        err_c = torch.abs(gt[mask] - pred[mask]).squeeze(-1)
        loss_conf = gamma * err_c * conf - alpha * torch.log(conf)
        loss_conf = loss_conf.mean() if loss_conf.numel() > 0 else (0.0 * pred).mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Gradient loss on residuals (optional)
    loss_grad = (0.0 * pred).mean()
    if gradient_loss_fn:
        B, S, H, W, C = pred.shape
        pred_b = pred.reshape(B * S, H, W, C)
        gt_b = gt.reshape(B * S, H, W, C)
        mask_b = mask.reshape(B * S, H, W)
        conf_b = None
        if pred_conf is not None and "conf" in gradient_loss_fn:
            conf_b = pred_conf.reshape(B * S, H, W)
        if "grad" in gradient_loss_fn:
            loss_grad = gradient_loss_multi_scale(pred_b, gt_b, mask_b, scales=3, conf=conf_b)

    return loss_conf, loss_grad, loss_reg


def camera_loss_single(pred_pose_enc: torch.Tensor, gt_pose_enc: torch.Tensor, *, loss_type: str = "l1"):
    """Compute translation/rotation/FoV loss for pose encodings (B,S,9)."""
    if loss_type == "l1":
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
        loss_T = loss_T.clamp(max=100).mean()
        loss_R = loss_R.mean()
        loss_FL = loss_FL.mean()
        return loss_T, loss_R, loss_FL
    if loss_type == "l2":
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1).mean()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1).mean()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1).mean()
        return loss_T, loss_R, loss_FL
    raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_camera_loss_from_extri_intri(
    pred_extr: torch.Tensor,
    pred_intr: torch.Tensor,
    gt_extr: torch.Tensor,
    gt_intr: torch.Tensor,
    image_hw: Tuple[int, int],
    *,
    loss_type: str = "l1",
    weight_trans: float = 1.0,
    weight_rot: float = 1.0,
    weight_focal: float = 0.5,
) -> Dict[str, torch.Tensor]:
    pred_pose = extri_intri_to_pose_encoding(pred_extr, pred_intr, image_hw)
    gt_pose = extri_intri_to_pose_encoding(gt_extr, gt_intr, image_hw)
    loss_T, loss_R, loss_FL = camera_loss_single(pred_pose, gt_pose, loss_type=loss_type)
    total = loss_T * weight_trans + loss_R * weight_rot + loss_FL * weight_focal
    return {"loss_camera": total, "loss_T": loss_T, "loss_R": loss_R, "loss_FL": loss_FL}


class DA3MultitaskDistillLoss(torch.nn.Module):
    """DA3 output-level loss (camera + depth)."""

    def __init__(self, camera: Optional[dict] = None, depth: Optional[dict] = None, **kwargs):
        super().__init__()
        self.camera = camera or {}
        self.depth = depth or {}

    def forward(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=batch["images"].device)
        loss_dict: Dict[str, torch.Tensor] = {}

        # Camera loss (requires extrinsics/intrinsics for both pred and gt)
        if all(k in predictions for k in ("extrinsics", "intrinsics")) and all(
            k in batch for k in ("extrinsics", "intrinsics")
        ):
            image_hw = tuple(batch["images"].shape[-2:])
            cam_losses = compute_camera_loss_from_extri_intri(
                predictions["extrinsics"],
                predictions["intrinsics"],
                batch["extrinsics"],
                batch["intrinsics"],
                image_hw,
                loss_type=self.camera.get("loss_type", "l1"),
                weight_trans=self.camera.get("weight_trans", 1.0),
                weight_rot=self.camera.get("weight_rot", 1.0),
                weight_focal=self.camera.get("weight_focal", 0.5),
            )
            cam_weight = float(self.camera.get("weight", 1.0))
            total_loss = total_loss + cam_weight * cam_losses["loss_camera"]
            loss_dict.update(cam_losses)

        # Depth loss
        if "depth" in predictions and "depths" in batch:
            pred_depth = predictions["depth"]
            pred_conf = predictions.get("depth_conf", None)
            gt_depth = batch["depths"]
            mask = batch.get("point_masks", None)
            if mask is None:
                mask = torch.ones_like(gt_depth, dtype=torch.bool)

            loss_conf, loss_grad, loss_reg = regression_depth_loss(
                pred_depth,
                gt_depth,
                mask,
                pred_conf=pred_conf,
                gradient_loss_fn=self.depth.get("gradient_loss_fn", None),
                gamma=float(self.depth.get("gamma", 1.0)),
                alpha=float(self.depth.get("alpha", 0.2)),
                valid_range=float(self.depth.get("valid_range", -1.0)),
            )
            depth_weight = float(self.depth.get("weight", 1.0))
            total_loss = total_loss + depth_weight * (loss_conf + loss_reg + loss_grad)
            loss_dict.update(
                {
                    "loss_conf_depth": loss_conf,
                    "loss_reg_depth": loss_reg,
                    "loss_grad_depth": loss_grad,
                }
            )

        loss_dict["objective"] = total_loss
        return loss_dict


@torch.no_grad()
def construct_da3_gt_batch(
    teacher_preds: Dict[str, torch.Tensor],
    student_images: torch.Tensor,
    student_frame_indices: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Build a GT batch dict from teacher predictions for student frames."""
    student_frame_indices = student_frame_indices or [0, 2, 4, 6]
    B, S, C, H, W = student_images.shape

    depth = teacher_preds.get("depth", None)
    extr = teacher_preds.get("extrinsics", None)
    intr = teacher_preds.get("intrinsics", None)
    if depth is None:
        raise KeyError("teacher_preds must contain 'depth' for output distillation")

    depth_sel = depth[:, student_frame_indices]
    extr_sel = extr[:, student_frame_indices] if extr is not None else None
    intr_sel = intr[:, student_frame_indices] if intr is not None else None

    dH, dW = depth_sel.shape[-2:]
    point_masks = torch.ones(B, len(student_frame_indices), dH, dW, dtype=torch.bool, device=student_images.device)

    batch = {
        "images": student_images,
        "depths": depth_sel.detach(),
        "point_masks": point_masks,
    }
    if extr_sel is not None:
        batch["extrinsics"] = extr_sel.detach()
    if intr_sel is not None:
        batch["intrinsics"] = intr_sel.detach()
    return batch
