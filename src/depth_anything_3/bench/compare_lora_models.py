# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compare 4 model configurations for camera token features and pose metrics:

1. Student 4-frame (original): frames [0,2,4,6] from 8-frame sample, direct pass
2. Student LoRA: frames [0,2,4,6] from 8-frame sample, with LoRA model
3. Teacher 8-frame: all 8 frames through original model
4. Teacher 8→4 extract: 8 frames encode, extract [0,2,4,6] features

Frame sampling strategy (matching benchmark):
- Sample 8 frames from scene using random.seed(42) + shuffle + sort
- Student uses even indices [0,2,4,6] from the 8 frames
- Teacher uses all 8 frames
- Reference view: "first" strategy

Pose metrics are loaded from saved benchmark results, not recomputed.

Supports multiple datasets:
- eth3d: ETH3D multiview dataset
- scannetpp: ScanNet++ indoor dataset
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict as AdictDict
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.registries import MV_REGISTRY
from depth_anything_3.cfg import load_config
from depth_anything_3.utils.io.input_processor import InputProcessor

# PEFT imports for LoRA
from peft import PeftModel


class ModelComparator:
    """
    Compare camera tokens and pose metrics across 4 model configurations.

    Supports multiple datasets:
    - eth3d: ETH3D multiview dataset
    - scannetpp: ScanNet++ indoor dataset
    """

    FRAME_INDICES = [0, 2, 4, 6]  # Even indices from 8-frame sample

    def __init__(
        self,
        model_path: str,
        lora_path: str,
        dataset_name: str = "eth3d",
        output_dir: str = "./analysis_output",
        pose_results_dir: str = "./workspace",
        debug: bool = False,
    ):
        self.model_path = model_path
        self.lora_path = lora_path
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.pose_results_dir = pose_results_dir
        self.debug = debug

        os.makedirs(output_dir, exist_ok=True)

        # Initialize dataset based on name
        self.dataset = MV_REGISTRY.get(dataset_name)()
        self.input_processor = InputProcessor()

        # Models (lazy loaded)
        self._original_model = None
        self._lora_model = None

        # Load saved pose metrics
        self.pose_metrics = self._load_pose_metrics()

    def _load_pose_metrics(self) -> Dict:
        """Load pose metrics from saved benchmark results."""
        metrics = {}

        # Define paths based on dataset
        dataset_prefix = self.dataset_name

        # Try to load 4-view student results
        student_path = os.path.join(
            self.pose_results_dir, f"evaluation_{dataset_prefix}_4views/metric_results/{dataset_prefix}_pose.json"
        )
        if os.path.exists(student_path):
            with open(student_path) as f:
                metrics['student_4frame'] = json.load(f)
            print(f"[INFO] Loaded student 4-view pose metrics from {student_path}")

        # Try to load 4-view LoRA results
        lora_path = os.path.join(
            self.pose_results_dir, f"eval_{dataset_prefix}_lora_4views/metric_results/{dataset_prefix}_pose.json"
        )
        if os.path.exists(lora_path):
            with open(lora_path) as f:
                metrics['student_lora'] = json.load(f)
            print(f"[INFO] Loaded LoRA 4-view pose metrics from {lora_path}")

        # Try to load 8-view teacher results
        teacher_path = os.path.join(
            self.pose_results_dir, f"evaluation_{dataset_prefix}_8views/metric_results/{dataset_prefix}_pose.json"
        )
        if os.path.exists(teacher_path):
            with open(teacher_path) as f:
                metrics['teacher_8frame'] = json.load(f)
            print(f"[INFO] Loaded teacher 8-view pose metrics from {teacher_path}")

        return metrics

    @property
    def original_model(self):
        """Lazy load original model."""
        if self._original_model is None:
            print(f"[INFO] Loading original model from {self.model_path}")
            self._original_model = DepthAnything3.from_pretrained(self.model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._original_model = self._original_model.to(device)
            self._original_model.eval()
        return self._original_model

    @property
    def lora_model(self):
        """Lazy load LoRA model."""
        if self._lora_model is None:
            print(f"[INFO] Loading LoRA model from {self.model_path} + {self.lora_path}")
            self._lora_model = DepthAnything3.from_pretrained(self.model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._lora_model = self._lora_model.to(device)

            # Load LoRA weights
            self._load_lora_weights(self._lora_model, self.lora_path)
            self._lora_model.eval()
        return self._lora_model

    def _load_lora_weights(self, model: DepthAnything3, lora_path: str) -> None:
        """Load PEFT LoRA weights into the model."""
        peft_path = lora_path.replace('.pt', '_peft')

        # Get backbone
        backbone = model.model.backbone.pretrained

        # Load PEFT adapter
        if os.path.exists(peft_path):
            print(f"[INFO] Loading PEFT adapter from {peft_path}")
            model.model.backbone.pretrained = PeftModel.from_pretrained(
                backbone, peft_path
            )
            print("[INFO] PEFT LoRA weights loaded")

        # Load camera token if present
        if os.path.exists(lora_path):
            state_dict = torch.load(lora_path, map_location='cpu')
            if 'camera_token' in state_dict:
                peft_model = model.model.backbone.pretrained
                # Navigate to the base model to find camera_token
                if hasattr(peft_model, 'base_model'):
                    base = peft_model.base_model
                    if hasattr(base, 'model'):
                        base = base.model
                    if hasattr(base, 'camera_token'):
                        base.camera_token.data.copy_(state_dict['camera_token'].to(base.camera_token.device))
                        print("[INFO] Camera token loaded")

    def _sample_frames(self, scene_data: AdictDict, num_frames: int = 8) -> AdictDict:
        """
        Sample frames deterministically using the same logic as the benchmark evaluator.

        Uses random.shuffle with seed 42, matching evaluator.py _sample_frames().
        """
        total_frames = len(scene_data.image_files)
        if total_frames <= num_frames:
            return scene_data

        # Use same sampling logic as evaluator.py
        random.seed(42)
        indices = list(range(total_frames))
        random.shuffle(indices)
        sampled_indices = sorted(indices[:num_frames])

        sampled = AdictDict()
        sampled.image_files = [scene_data.image_files[i] for i in sampled_indices]
        sampled.extrinsics = scene_data.extrinsics[sampled_indices]
        sampled.intrinsics = scene_data.intrinsics[sampled_indices]
        sampled.aux = scene_data.aux
        return sampled

    def _preprocess_images(self, scene_data: AdictDict, model) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and preprocess images."""
        imgs_tensor, extrinsics, intrinsics = self.input_processor(
            scene_data.image_files,
            scene_data.extrinsics,
            scene_data.intrinsics,
            process_res=504,
            process_res_method="upper_bound_resize",
            num_workers=4,
            print_progress=False,
        )
        device = model._get_model_device()
        imgs_tensor = imgs_tensor.to(device).float()
        extrinsics = extrinsics.to(device).float()
        intrinsics = intrinsics.to(device).float()

        if imgs_tensor.dim() == 4:
            imgs_tensor = imgs_tensor[None]
            extrinsics = extrinsics[None]
            intrinsics = intrinsics[None]

        return imgs_tensor, extrinsics, intrinsics

    def extract_camera_tokens(
        self,
        images_8: torch.Tensor,
        extrinsics_8: torch.Tensor,
        intrinsics_8: torch.Tensor,
    ) -> Dict[str, AdictDict]:
        """
        Extract camera tokens for all 4 configurations.

        Strategy:
        - Sample 8 frames from scene
        - Student/LoRA use frames [0,2,4,6] (even indices)
        - Teacher uses all 8 frames
        - Teacher 8→4 extracts [0,2,4,6] from 8-frame features

        Args:
            images_8: 8-frame images [B, 8, 3, H, W]
            extrinsics_8: GT extrinsics for 8 frames
            intrinsics_8: GT intrinsics for 8 frames

        Returns dict with camera tokens for each configuration
        """
        results = {}
        device = images_8.device
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Get network references
        net_orig = self.original_model.model
        net_lora = self.lora_model.model

        # Extract 4-frame subset (even indices)
        images_4 = images_8[:, self.FRAME_INDICES]
        ext_4 = extrinsics_8[:, self.FRAME_INDICES]
        int_4 = intrinsics_8[:, self.FRAME_INDICES]

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):

                # 1. Student 4-frame (original model) - uses [0,2,4,6]
                feats_student_4, _, H, W = net_orig.forward_backbone_only(
                    images_4,
                    extrinsics=None,
                    intrinsics=None,
                    ref_view_strategy="first",
                )
                results['student_4frame'] = AdictDict(
                    feats=feats_student_4,
                    gt_extrinsics=ext_4,
                    gt_intrinsics=int_4,
                )

                # 2. Student LoRA (4-frame with LoRA model)
                feats_lora, _, H, W = net_lora.forward_backbone_only(
                    images_4,
                    extrinsics=None,
                    intrinsics=None,
                    ref_view_strategy="first",
                )
                results['student_lora'] = AdictDict(
                    feats=feats_lora,
                    gt_extrinsics=ext_4,
                    gt_intrinsics=int_4,
                )

                # 3. Teacher 8-frame (original model) - uses all 8
                feats_teacher_8, _, H8, W8 = net_orig.forward_backbone_only(
                    images_8,
                    extrinsics=None,
                    intrinsics=None,
                    ref_view_strategy="first",
                )
                results['teacher_8frame'] = AdictDict(
                    feats=feats_teacher_8,
                    gt_extrinsics=extrinsics_8,
                    gt_intrinsics=intrinsics_8,
                )

                # 4. Teacher 8→4 extract (8-frame encode, extract [0,2,4,6])
                feats_8to4 = []
                for patch_feat, cam_token in feats_teacher_8:
                    feats_8to4.append((
                        patch_feat[:, self.FRAME_INDICES],
                        cam_token[:, self.FRAME_INDICES]
                    ))
                results['teacher_8to4'] = AdictDict(
                    feats=feats_8to4,
                    gt_extrinsics=ext_4,
                    gt_intrinsics=int_4,
                )

        return results

    def compute_camera_token_stats(
        self,
        results: Dict[str, AdictDict],
    ) -> Dict[str, Dict]:
        """
        Compute camera token statistics between configurations.

        Comparisons:
        - student_4frame vs teacher_8to4 (baseline gap)
        - student_lora vs teacher_8to4 (lora improvement)
        - student_lora vs student_4frame (lora effect)
        """
        stats = {}

        # Get last layer camera tokens (full 3072-dim)
        # feats is a list of (patch_feat, cam_token) tuples
        cam_student_4 = results['student_4frame'].feats[-1][1].float()  # [B, 4, 3072]
        cam_lora = results['student_lora'].feats[-1][1].float()
        cam_teacher_8 = results['teacher_8frame'].feats[-1][1].float()  # [B, 8, 3072]
        cam_8to4 = results['teacher_8to4'].feats[-1][1].float()  # [B, 4, 3072]

        # Flatten for statistics (combine batch and sequence dims)
        cam_student_4_flat = cam_student_4.reshape(-1, cam_student_4.shape[-1])
        cam_lora_flat = cam_lora.reshape(-1, cam_lora.shape[-1])
        cam_8to4_flat = cam_8to4.reshape(-1, cam_8to4.shape[-1])

        # 1. Student 4-frame vs Teacher 8→4 (baseline gap)
        stats['student_vs_teacher'] = self._compute_token_diff(
            cam_student_4_flat, cam_8to4_flat, "student_4frame", "teacher_8to4"
        )

        # 2. LoRA vs Teacher 8→4 (lora improvement)
        stats['lora_vs_teacher'] = self._compute_token_diff(
            cam_lora_flat, cam_8to4_flat, "student_lora", "teacher_8to4"
        )

        # 3. LoRA vs Student 4-frame (lora effect)
        stats['lora_vs_student'] = self._compute_token_diff(
            cam_lora_flat, cam_student_4_flat, "student_lora", "student_4frame"
        )

        # Store raw tokens for visualization
        stats['tokens'] = {
            'student_4frame': cam_student_4_flat.cpu().numpy(),
            'student_lora': cam_lora_flat.cpu().numpy(),
            'teacher_8to4': cam_8to4_flat.cpu().numpy(),
        }

        # Per-frame analysis
        stats['per_frame'] = {}
        for i in range(4):
            frame_student = cam_student_4[0, i].float()
            frame_lora = cam_lora[0, i].float()
            frame_teacher = cam_8to4[0, i].float()

            stats['per_frame'][f'frame_{i}'] = {
                'student_vs_teacher_mse': F.mse_loss(frame_student, frame_teacher).item(),
                'lora_vs_teacher_mse': F.mse_loss(frame_lora, frame_teacher).item(),
                'lora_vs_student_mse': F.mse_loss(frame_lora, frame_student).item(),
                'student_vs_teacher_cos': F.cosine_similarity(frame_student.unsqueeze(0), frame_teacher.unsqueeze(0)).item(),
                'lora_vs_teacher_cos': F.cosine_similarity(frame_lora.unsqueeze(0), frame_teacher.unsqueeze(0)).item(),
            }

        return stats

    def compute_patch_feature_stats(
        self,
        results: Dict[str, AdictDict],
    ) -> Dict[str, Dict]:
        """
        Compare [B, S, P, 3072] patch features between teacher and student.

        This analyzes the full patch-level features, not just camera tokens.
        Compares:
        - Student 4-frame patch features vs Teacher 8→4 patch features
        - LoRA patch features vs Teacher 8→4 patch features

        Returns per-patch cosine similarity, MSE, and spatial statistics.
        """
        stats = {}

        # Get last layer patch features (full 3072-dim)
        # feats is a list of (patch_feat, cam_token) tuples
        # patch_feat shape: [B, S, P, 3072] where P = H*W patches
        patch_student_4 = results['student_4frame'].feats[-1][0].float()  # [B, 4, P, 3072]
        patch_lora = results['student_lora'].feats[-1][0].float()  # [B, 4, P, 3072]
        patch_8to4 = results['teacher_8to4'].feats[-1][0].float()  # [B, 4, P, 3072]

        B, S, P, C = patch_student_4.shape

        # Store spatial dimensions for visualization
        stats['spatial_dims'] = {'B': B, 'S': S, 'P': P, 'C': C}

        # 1. Per-patch cosine similarity [B, S, P]
        cos_sim_student = F.cosine_similarity(patch_student_4, patch_8to4, dim=-1)
        cos_sim_lora = F.cosine_similarity(patch_lora, patch_8to4, dim=-1)

        # 2. Per-patch MSE [B, S, P]
        mse_student = F.mse_loss(patch_student_4, patch_8to4, reduction='none').mean(dim=-1)
        mse_lora = F.mse_loss(patch_lora, patch_8to4, reduction='none').mean(dim=-1)

        # 3. Student vs Teacher statistics
        stats['student_vs_teacher'] = {
            # Per-view mean cosine similarity [S]
            'cos_sim_per_view': cos_sim_student.mean(dim=(0, 2)).tolist(),  # [4]
            'cos_sim_mean': cos_sim_student.mean().item(),
            'cos_sim_std': cos_sim_student.std().item(),
            'cos_sim_min': cos_sim_student.min().item(),
            'cos_sim_max': cos_sim_student.max().item(),
            # Per-view mean MSE [S]
            'mse_per_view': mse_student.mean(dim=(0, 2)).tolist(),  # [4]
            'mse_mean': mse_student.mean().item(),
            'mse_std': mse_student.std().item(),
        }

        # 4. LoRA vs Teacher statistics
        stats['lora_vs_teacher'] = {
            'cos_sim_per_view': cos_sim_lora.mean(dim=(0, 2)).tolist(),
            'cos_sim_mean': cos_sim_lora.mean().item(),
            'cos_sim_std': cos_sim_lora.std().item(),
            'cos_sim_min': cos_sim_lora.min().item(),
            'cos_sim_max': cos_sim_lora.max().item(),
            'mse_per_view': mse_lora.mean(dim=(0, 2)).tolist(),
            'mse_mean': mse_lora.mean().item(),
            'mse_std': mse_lora.std().item(),
        }

        # 5. First half (local) vs second half (global) analysis
        half = C // 2  # 1536
        for name, (a, b) in [
            ('student_vs_teacher', (patch_student_4, patch_8to4)),
            ('lora_vs_teacher', (patch_lora, patch_8to4))
        ]:
            stats[name]['cos_sim_local'] = F.cosine_similarity(
                a[..., :half], b[..., :half], dim=-1
            ).mean().item()
            stats[name]['cos_sim_global'] = F.cosine_similarity(
                a[..., half:], b[..., half:], dim=-1
            ).mean().item()
            stats[name]['mse_local'] = F.mse_loss(
                a[..., :half], b[..., :half]
            ).item()
            stats[name]['mse_global'] = F.mse_loss(
                a[..., half:], b[..., half:]
            ).item()

        # 6. Improvement map (lora - student similarity)
        improvement = cos_sim_lora - cos_sim_student  # [B, S, P]
        stats['improvement'] = {
            'mean': improvement.mean().item(),
            'std': improvement.std().item(),
            'min': improvement.min().item(),
            'max': improvement.max().item(),
            'positive_ratio': (improvement > 0).float().mean().item(),  # % patches improved
            'per_view': improvement.mean(dim=(0, 2)).tolist(),  # [4]
        }

        # Store raw tensors for visualization (on CPU to save GPU memory)
        stats['tensors'] = {
            'cos_sim_student': cos_sim_student.cpu(),  # [B, S, P]
            'cos_sim_lora': cos_sim_lora.cpu(),  # [B, S, P]
            'improvement': improvement.cpu(),  # [B, S, P]
            'patch_student': patch_student_4.cpu(),  # [B, S, P, 3072]
            'patch_lora': patch_lora.cpu(),  # [B, S, P, 3072]
            'patch_teacher': patch_8to4.cpu(),  # [B, S, P, 3072]
        }

        return stats

    def visualize_patch_features_on_images(
        self,
        images: torch.Tensor,  # [B, S, 3, H, W]
        patch_stats: Dict,
        scene_name: str,
        H_patches: int,
        W_patches: int,
    ):
        """
        Visualize patch feature differences overlaid on original images.

        Creates 3 types of visualizations:
        1. Cosine similarity heatmap - Red=low, Green=high similarity to teacher
        2. PCA RGB coloring - 3072D → 3D RGB for feature distribution patterns
        3. Improvement map - Shows where LoRA helps (blue) vs hurts (red)

        Args:
            images: Original images [B, S, 3, H, W]
            patch_stats: Output from compute_patch_feature_stats()
            scene_name: Name of the scene for saving
            H_patches, W_patches: Patch grid dimensions
        """
        import cv2
        from sklearn.decomposition import PCA

        B, S, _, H_img, W_img = images.shape
        tensors = patch_stats['tensors']

        # Get similarity maps [B, S, P] and reshape to spatial [B, S, H, W]
        cos_sim_student = tensors['cos_sim_student'].reshape(B, S, H_patches, W_patches)
        cos_sim_lora = tensors['cos_sim_lora'].reshape(B, S, H_patches, W_patches)
        improvement = tensors['improvement'].reshape(B, S, H_patches, W_patches)

        # Get patch features for PCA [B, S, P, 3072]
        patch_student = tensors['patch_student']
        patch_lora = tensors['patch_lora']
        patch_teacher = tensors['patch_teacher']

        # Compute PCA for RGB visualization (fit on teacher, transform all)
        # Reshape to [B*S*P, 3072] for PCA
        teacher_flat = patch_teacher.reshape(-1, patch_teacher.shape[-1]).numpy()
        student_flat = patch_student.reshape(-1, patch_student.shape[-1]).numpy()
        lora_flat = patch_lora.reshape(-1, patch_lora.shape[-1]).numpy()

        # Fit PCA on teacher features
        pca = PCA(n_components=3)
        pca.fit(teacher_flat)

        # Transform all to 3D
        teacher_pca = pca.transform(teacher_flat).reshape(B, S, H_patches, W_patches, 3)
        student_pca = pca.transform(student_flat).reshape(B, S, H_patches, W_patches, 3)
        lora_pca = pca.transform(lora_flat).reshape(B, S, H_patches, W_patches, 3)

        # Normalize PCA to [0, 1] for visualization
        def normalize_pca(x):
            x_min, x_max = x.min(), x.max()
            if x_max - x_min > 1e-6:
                return (x - x_min) / (x_max - x_min)
            return x * 0 + 0.5

        teacher_pca = normalize_pca(teacher_pca)
        student_pca = normalize_pca(student_pca)
        lora_pca = normalize_pca(lora_pca)

        # Create visualizations for each frame
        for b in range(B):
            for s in range(S):
                fig, axes = plt.subplots(2, 4, figsize=(24, 12))

                # Get original image and normalize to [0, 1]
                img = images[b, s].permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                # === Row 1: Original + Cosine similarity heatmaps + Improvement ===

                # Original image
                axes[0, 0].imshow(img)
                axes[0, 0].set_title(f'Original Frame {s}', fontsize=12)
                axes[0, 0].axis('off')

                # Student cosine similarity heatmap
                sim_map = cos_sim_student[b, s].numpy()
                sim_resized = cv2.resize(sim_map, (img.shape[1], img.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                im1 = axes[0, 1].imshow(sim_resized, cmap='RdYlGn', vmin=0.85, vmax=1.0)
                axes[0, 1].set_title(f'Student Cos Sim: {sim_map.mean():.4f}', fontsize=12)
                plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
                axes[0, 1].axis('off')

                # LoRA cosine similarity heatmap
                sim_map_lora = cos_sim_lora[b, s].numpy()
                sim_resized_lora = cv2.resize(sim_map_lora, (img.shape[1], img.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                im2 = axes[0, 2].imshow(sim_resized_lora, cmap='RdYlGn', vmin=0.85, vmax=1.0)
                axes[0, 2].set_title(f'LoRA Cos Sim: {sim_map_lora.mean():.4f}', fontsize=12)
                plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
                axes[0, 2].axis('off')

                # Improvement map (LoRA - Student)
                imp_map = improvement[b, s].numpy()
                imp_resized = cv2.resize(imp_map, (img.shape[1], img.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                im3 = axes[0, 3].imshow(imp_resized, cmap='RdBu', vmin=-0.05, vmax=0.05)
                axes[0, 3].set_title(f'Improvement (LoRA-Student): {imp_map.mean():.4f}', fontsize=12)
                plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
                axes[0, 3].axis('off')

                # === Row 2: PCA RGB visualizations ===

                # Original (repeated for reference)
                axes[1, 0].imshow(img)
                axes[1, 0].set_title('Original', fontsize=12)
                axes[1, 0].axis('off')

                # Teacher PCA
                teacher_rgb = teacher_pca[b, s]
                teacher_rgb_resized = cv2.resize(teacher_rgb, (img.shape[1], img.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
                axes[1, 1].imshow(teacher_rgb_resized)
                axes[1, 1].set_title('Teacher Features (PCA)', fontsize=12)
                axes[1, 1].axis('off')

                # Student PCA
                student_rgb = student_pca[b, s]
                student_rgb_resized = cv2.resize(student_rgb, (img.shape[1], img.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
                axes[1, 2].imshow(student_rgb_resized)
                axes[1, 2].set_title('Student Features (PCA)', fontsize=12)
                axes[1, 2].axis('off')

                # LoRA PCA
                lora_rgb = lora_pca[b, s]
                lora_rgb_resized = cv2.resize(lora_rgb, (img.shape[1], img.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                axes[1, 3].imshow(lora_rgb_resized)
                axes[1, 3].set_title('LoRA Features (PCA)', fontsize=12)
                axes[1, 3].axis('off')

                plt.suptitle(f'{scene_name} - Frame {s} | Patch Features Analysis',
                            fontsize=14, fontweight='bold')
                plt.tight_layout()

                save_path = os.path.join(
                    self.output_dir, f'{scene_name}_frame{s}_patch_features.png'
                )
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

        print(f"  Saved patch feature visualizations to {self.output_dir}/{scene_name}_frame*_patch_features.png")

    def _compute_token_diff(
        self,
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
        name_a: str,
        name_b: str,
    ) -> Dict:
        """Compute difference statistics between two token sets."""
        # MSE
        mse = F.mse_loss(tokens_a, tokens_b, reduction='none').mean(dim=-1)

        # Cosine similarity
        cos_sim = F.cosine_similarity(tokens_a, tokens_b, dim=-1)

        # L2 norm of difference
        diff_norm = (tokens_a - tokens_b).norm(dim=-1)

        # Per-dimension analysis
        diff = (tokens_a - tokens_b).abs()

        # First half vs second half analysis (local vs global)
        half_dim = tokens_a.shape[-1] // 2
        mse_first_half = F.mse_loss(tokens_a[..., :half_dim], tokens_b[..., :half_dim]).item()
        mse_second_half = F.mse_loss(tokens_a[..., half_dim:], tokens_b[..., half_dim:]).item()

        return {
            'comparison': f"{name_a} vs {name_b}",
            'mse_mean': mse.mean().item(),
            'mse_std': mse.std().item(),
            'cos_sim_mean': cos_sim.mean().item(),
            'cos_sim_std': cos_sim.std().item(),
            'diff_norm_mean': diff_norm.mean().item(),
            'diff_norm_std': diff_norm.std().item(),
            'mse_first_half': mse_first_half,
            'mse_second_half': mse_second_half,
            'diff_per_dim_mean': diff.mean(dim=0).cpu().numpy(),
        }

    def get_scene_pose_metrics(self, scene: str) -> Dict[str, Dict]:
        """Get saved pose metrics for a scene."""
        metrics = {}

        for config in ['student_4frame', 'student_lora', 'teacher_8frame']:
            if config in self.pose_metrics and scene in self.pose_metrics[config]:
                m = self.pose_metrics[config][scene]
                metrics[config] = {
                    'auc@3': m.get('auc03', 0),
                    'auc@5': m.get('auc05', 0),
                    'auc@15': m.get('auc15', 0),
                    'auc@30': m.get('auc30', 0),
                }

        return metrics

    def visualize_comparison(
        self,
        token_stats: Dict,
        pose_metrics: Dict,
        scene_name: str,
    ):
        """Create visualization comparing all configurations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Camera token MSE comparison
        ax = axes[0, 0]
        comparisons = ['student_vs_teacher', 'lora_vs_teacher', 'lora_vs_student']
        labels = ['Student vs Teacher', 'LoRA vs Teacher', 'LoRA vs Student']
        mse_vals = [token_stats[c]['mse_mean'] for c in comparisons]
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        bars = ax.bar(labels, mse_vals, color=colors)
        ax.set_ylabel('MSE')
        ax.set_title('Camera Token MSE Comparison')
        for bar, val in zip(bars, mse_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        # 2. First half vs Second half MSE
        ax = axes[0, 1]
        x = np.arange(2)
        width = 0.25
        for i, comp in enumerate(comparisons):
            first_half = token_stats[comp]['mse_first_half']
            second_half = token_stats[comp]['mse_second_half']
            ax.bar(x + i*width, [first_half, second_half], width, label=labels[i], color=colors[i])
        ax.set_ylabel('MSE')
        ax.set_title('First Half (local) vs Second Half (global) MSE')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['First Half [0:1536]', 'Second Half [1536:3072]'])
        ax.legend()

        # 3. Pose AUC comparison (from saved metrics)
        ax = axes[0, 2]
        if pose_metrics:
            configs = list(pose_metrics.keys())
            config_labels = [c.replace('_', ' ').title() for c in configs]
            x = np.arange(len(configs))
            width = 0.2

            thresholds = [3, 5, 15, 30]
            for i, thresh in enumerate(thresholds):
                vals = [pose_metrics[c].get(f'auc@{thresh}', 0) for c in configs]
                ax.bar(x + i*width, vals, width, label=f'AUC@{thresh}')

            ax.set_ylabel('AUC')
            ax.set_title('Pose AUC (from saved benchmark)')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(config_labels, rotation=15)
            ax.legend()
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'No saved pose metrics', ha='center', va='center')
            ax.set_title('Pose AUC (no data)')

        # 4. Per-dimension difference heatmap (Student vs Teacher)
        ax = axes[1, 0]
        diff_student = token_stats['student_vs_teacher']['diff_per_dim_mean']
        diff_lora = token_stats['lora_vs_teacher']['diff_per_dim_mean']

        # Show top 100 dimensions with highest difference
        top_dims = np.argsort(diff_student)[-100:]

        im = ax.imshow([diff_student[top_dims], diff_lora[top_dims]],
                       aspect='auto', cmap='hot')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Student', 'LoRA'])
        ax.set_xlabel('Top 100 Diff Dimensions')
        ax.set_title('Per-Dimension Difference (vs Teacher)')
        plt.colorbar(im, ax=ax)

        # 5. Per-frame MSE analysis
        ax = axes[1, 1]
        if 'per_frame' in token_stats:
            frames = list(token_stats['per_frame'].keys())
            student_mse = [token_stats['per_frame'][f]['student_vs_teacher_mse'] for f in frames]
            lora_mse = [token_stats['per_frame'][f]['lora_vs_teacher_mse'] for f in frames]

            x = np.arange(len(frames))
            width = 0.35
            ax.bar(x - width/2, student_mse, width, label='Student vs Teacher', color='#ff6b6b')
            ax.bar(x + width/2, lora_mse, width, label='LoRA vs Teacher', color='#4ecdc4')
            ax.set_ylabel('MSE')
            ax.set_title('Per-Frame Camera Token MSE')
            ax.set_xticks(x)
            ax.set_xticklabels(['Frame 0', 'Frame 2', 'Frame 4', 'Frame 6'])
            ax.legend()

        # 6. Gap closed analysis
        ax = axes[1, 2]
        if pose_metrics and 'student_4frame' in pose_metrics and 'teacher_8frame' in pose_metrics:
            gap_closed = {}
            thresholds = [3, 5, 15, 30]
            for thresh in thresholds:
                student_auc = pose_metrics.get('student_4frame', {}).get(f'auc@{thresh}', 0)
                lora_auc = pose_metrics.get('student_lora', {}).get(f'auc@{thresh}', 0)
                teacher_auc = pose_metrics.get('teacher_8frame', {}).get(f'auc@{thresh}', 0)

                gap = teacher_auc - student_auc
                if abs(gap) > 1e-6:
                    closed = (lora_auc - student_auc) / gap * 100
                else:
                    closed = 0
                gap_closed[thresh] = closed

            bars = ax.bar([f'AUC@{t}' for t in thresholds],
                          [gap_closed[t] for t in thresholds],
                          color=['#2ecc71' if g > 0 else '#e74c3c' for g in gap_closed.values()])
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(y=100, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_ylabel('Gap Closed (%)')
            ax.set_title('LoRA Gap Closed (Student→Teacher)')

            for bar, val in zip(bars, gap_closed.values()):
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + (5 if val > 0 else -15),
                       f'{val:.1f}%', ha='center', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Need both student & teacher metrics', ha='center', va='center')
            ax.set_title('Gap Closed (no data)')

        plt.suptitle(f'Model Comparison: {scene_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'model_comparison_{scene_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved visualization to {self.output_dir}/model_comparison_{scene_name}.png")

    def run_comparison(self, scenes: Optional[List[str]] = None):
        """Run full comparison on specified scenes."""
        all_scenes = self.dataset.SCENES
        if scenes:
            all_scenes = [s for s in all_scenes if s in scenes]

        all_results = []

        for scene in tqdm(all_scenes, desc="Comparing models"):
            print(f"\n[COMPARE] Processing {scene}")

            # Get scene data
            scene_data = self.dataset.get_data(scene)

            # Sample 8 frames using benchmark's sampling logic
            sampled_data_8 = self._sample_frames(scene_data, num_frames=8)

            if len(sampled_data_8.image_files) < 8:
                print(f"  [SKIP] {scene}: not enough frames for 8-frame")
                continue

            # Preprocess 8 frames
            images_8, extrinsics_8, intrinsics_8 = self._preprocess_images(
                sampled_data_8, self.original_model
            )

            # Extract camera tokens for all configurations
            results = self.extract_camera_tokens(images_8, extrinsics_8, intrinsics_8)

            # Compute camera token statistics
            token_stats = self.compute_camera_token_stats(results)

            # Compute patch feature statistics [B, S, P, 3072]
            patch_stats = self.compute_patch_feature_stats(results)

            # Get saved pose metrics for this scene
            pose_metrics = self.get_scene_pose_metrics(scene)

            # Print summary
            self._print_scene_summary(scene, token_stats, patch_stats, pose_metrics)

            # Visualize camera token comparison
            self.visualize_comparison(token_stats, pose_metrics, scene)

            # Visualize patch features on images
            # Get 4-frame images for visualization
            images_4 = images_8[:, self.FRAME_INDICES]
            # Infer patch grid dimensions from P
            P = patch_stats['spatial_dims']['P']
            # Images are processed at 504, so patches are 504/14 = 36
            H_patches = W_patches = int(np.sqrt(P))
            if H_patches * W_patches != P:
                # Try common aspect ratios
                for h in [36, 32, 28, 24]:
                    if P % h == 0:
                        H_patches = h
                        W_patches = P // h
                        break
            self.visualize_patch_features_on_images(
                images_4, patch_stats, scene, H_patches, W_patches
            )

            all_results.append((scene, token_stats, patch_stats, pose_metrics))

        # Print aggregate results
        self._print_aggregate_summary(all_results)

        return all_results

    def _print_scene_summary(
        self,
        scene: str,
        token_stats: Dict,
        patch_stats: Dict,
        pose_metrics: Dict,
    ):
        """Print summary for a single scene."""
        print(f"\n  === {scene} ===")

        print("\n  Camera Token Statistics [B, S, 3072]:")
        for key in ['student_vs_teacher', 'lora_vs_teacher', 'lora_vs_student']:
            stats = token_stats[key]
            print(f"    {stats['comparison']}:")
            print(f"      MSE: {stats['mse_mean']:.6f} ± {stats['mse_std']:.6f}")
            print(f"      Cosine Sim: {stats['cos_sim_mean']:.6f}")
            print(f"      MSE 1st half: {stats['mse_first_half']:.6f}, 2nd half: {stats['mse_second_half']:.6f}")

        print("\n  Patch Feature Statistics [B, S, P, 3072]:")
        dims = patch_stats['spatial_dims']
        print(f"    Dimensions: B={dims['B']}, S={dims['S']}, P={dims['P']}, C={dims['C']}")

        for key in ['student_vs_teacher', 'lora_vs_teacher']:
            stats = patch_stats[key]
            print(f"    {key}:")
            print(f"      Cosine Sim: {stats['cos_sim_mean']:.4f} ± {stats['cos_sim_std']:.4f} "
                  f"(min={stats['cos_sim_min']:.4f}, max={stats['cos_sim_max']:.4f})")
            print(f"      MSE: {stats['mse_mean']:.6f}")
            print(f"      Local (0:1536): cos={stats['cos_sim_local']:.4f}, mse={stats['mse_local']:.6f}")
            print(f"      Global (1536:3072): cos={stats['cos_sim_global']:.4f}, mse={stats['mse_global']:.6f}")
            print(f"      Per-view cos sim: {[f'{v:.4f}' for v in stats['cos_sim_per_view']]}")

        imp = patch_stats['improvement']
        print(f"    Improvement (LoRA - Student):")
        print(f"      Mean: {imp['mean']:.4f}, Std: {imp['std']:.4f}")
        print(f"      Positive ratio: {imp['positive_ratio']*100:.1f}% of patches improved")
        print(f"      Per-view: {[f'{v:.4f}' for v in imp['per_view']]}")

        if pose_metrics:
            print("\n  Pose AUC Metrics (from saved benchmark):")
            print(f"    {'Config':<20} {'AUC@3':>10} {'AUC@5':>10} {'AUC@15':>10} {'AUC@30':>10}")
            print("    " + "-" * 62)
            for config in ['student_4frame', 'student_lora', 'teacher_8frame']:
                if config in pose_metrics:
                    m = pose_metrics[config]
                    print(f"    {config:<20} {m['auc@3']:>10.4f} {m['auc@5']:>10.4f} "
                          f"{m['auc@15']:>10.4f} {m['auc@30']:>10.4f}")

    def _print_aggregate_summary(self, all_results: List):
        """Print aggregate summary across all scenes."""
        if not all_results:
            return

        print("\n" + "=" * 80)
        print("AGGREGATE RESULTS ACROSS ALL SCENES")
        print("=" * 80)

        # Collect token stats
        token_mse = {'student_vs_teacher': [], 'lora_vs_teacher': [], 'lora_vs_student': []}
        token_cos = {'student_vs_teacher': [], 'lora_vs_teacher': [], 'lora_vs_student': []}
        token_mse_first = {'student_vs_teacher': [], 'lora_vs_teacher': []}
        token_mse_second = {'student_vs_teacher': [], 'lora_vs_teacher': []}

        # Collect patch stats
        patch_cos = {'student_vs_teacher': [], 'lora_vs_teacher': []}
        patch_mse = {'student_vs_teacher': [], 'lora_vs_teacher': []}
        patch_cos_local = {'student_vs_teacher': [], 'lora_vs_teacher': []}
        patch_cos_global = {'student_vs_teacher': [], 'lora_vs_teacher': []}
        improvement_mean = []
        improvement_positive_ratio = []

        for scene, token_stats, patch_stats, pose_metrics in all_results:
            for key in token_mse:
                token_mse[key].append(token_stats[key]['mse_mean'])
                token_cos[key].append(token_stats[key]['cos_sim_mean'])
            for key in token_mse_first:
                token_mse_first[key].append(token_stats[key]['mse_first_half'])
                token_mse_second[key].append(token_stats[key]['mse_second_half'])

            # Patch stats
            for key in patch_cos:
                patch_cos[key].append(patch_stats[key]['cos_sim_mean'])
                patch_mse[key].append(patch_stats[key]['mse_mean'])
                patch_cos_local[key].append(patch_stats[key]['cos_sim_local'])
                patch_cos_global[key].append(patch_stats[key]['cos_sim_global'])

            improvement_mean.append(patch_stats['improvement']['mean'])
            improvement_positive_ratio.append(patch_stats['improvement']['positive_ratio'])

        # Print Camera Token stats
        print("\n  CAMERA TOKEN STATISTICS [B, S, 3072] (mean across scenes):")
        print(f"    {'Comparison':<25} {'MSE':>12} {'Cosine Sim':>12}")
        print("    " + "-" * 52)
        for key in token_mse:
            print(f"    {key:<25} {np.mean(token_mse[key]):>12.6f} {np.mean(token_cos[key]):>12.4f}")

        print("\n  CAMERA TOKEN MSE by Half:")
        print(f"    {'Comparison':<25} {'1st Half (local)':>18} {'2nd Half (global)':>18}")
        print("    " + "-" * 65)
        for key in token_mse_first:
            first_mean = np.mean(token_mse_first[key])
            second_mean = np.mean(token_mse_second[key])
            print(f"    {key:<25} {first_mean:>18.6f} {second_mean:>18.6f}")

        # Print Patch Feature stats
        print("\n  PATCH FEATURE STATISTICS [B, S, P, 3072] (mean across scenes):")
        print(f"    {'Comparison':<25} {'Cos Sim':>12} {'MSE':>12} {'Cos Local':>12} {'Cos Global':>12}")
        print("    " + "-" * 78)
        for key in patch_cos:
            print(f"    {key:<25} {np.mean(patch_cos[key]):>12.4f} {np.mean(patch_mse[key]):>12.6f} "
                  f"{np.mean(patch_cos_local[key]):>12.4f} {np.mean(patch_cos_global[key]):>12.4f}")

        print(f"\n  PATCH IMPROVEMENT (LoRA - Student):")
        print(f"    Mean improvement: {np.mean(improvement_mean):.4f} ± {np.std(improvement_mean):.4f}")
        print(f"    Positive ratio: {np.mean(improvement_positive_ratio)*100:.1f}% ± "
              f"{np.std(improvement_positive_ratio)*100:.1f}% of patches improved")

        # Print saved pose metrics summary
        if self.pose_metrics:
            print("\n  POSE AUC (from saved benchmark - mean across scenes):")
            print(f"    {'Config':<20} {'AUC@3':>10} {'AUC@5':>10} {'AUC@15':>10} {'AUC@30':>10}")
            print("    " + "-" * 62)
            for config in ['student_4frame', 'student_lora', 'teacher_8frame']:
                if config in self.pose_metrics and 'mean' in self.pose_metrics[config]:
                    m = self.pose_metrics[config]['mean']
                    print(f"    {config:<20} {m.get('auc03', 0):>10.4f} {m.get('auc05', 0):>10.4f} "
                          f"{m.get('auc15', 0):>10.4f} {m.get('auc30', 0):>10.4f}")

            # Compute gap closed
            if all(c in self.pose_metrics for c in ['student_4frame', 'student_lora', 'teacher_8frame']):
                print("\n  GAP CLOSED (LoRA improvement towards Teacher):")
                print(f"    {'Metric':<15} {'Gap Closed %':>15}")
                print("    " + "-" * 32)
                for metric in ['auc03', 'auc05', 'auc15', 'auc30']:
                    s = self.pose_metrics['student_4frame']['mean'].get(metric, 0)
                    l = self.pose_metrics['student_lora']['mean'].get(metric, 0)
                    t = self.pose_metrics['teacher_8frame']['mean'].get(metric, 0)
                    gap = t - s
                    if abs(gap) > 1e-6:
                        closed = (l - s) / gap * 100
                    else:
                        closed = 0
                    print(f"    {metric:<15} {closed:>14.1f}%")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare LoRA model with original")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--dataset", type=str, default="eth3d",
                       choices=["eth3d", "scannetpp"],
                       help="Dataset to use for comparison")
    parser.add_argument("--lora_path", type=str,
                       default="/home/22097845d/Depth-Anything-3/checkpoints/lora_v2/best_lora.pt",
                       help="Path to LoRA weights")
    parser.add_argument("--scenes", type=str, nargs="+", help="Specific scenes to analyze")
    parser.add_argument("--output_dir", type=str, default="./analysis_output",
                       help="Output directory for visualizations")
    parser.add_argument("--pose_results_dir", type=str, default="./workspace",
                       help="Directory containing saved benchmark pose results")
    args, remaining = parser.parse_known_args()

    # Load config
    config_path = args.config or os.path.join(
        os.path.dirname(__file__), "configs/eval_eth3d_multiview.yaml"
    )
    config = load_config(config_path, argv=remaining)

    comparator = ModelComparator(
        model_path=config.model.path,
        lora_path=args.lora_path,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        pose_results_dir=args.pose_results_dir,
        debug=True,
    )

    comparator.run_comparison(scenes=args.scenes)


if __name__ == "__main__":
    main()
