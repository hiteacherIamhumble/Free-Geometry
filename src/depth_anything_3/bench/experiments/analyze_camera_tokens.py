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
Camera Token Analysis for Frame Extraction Experiments.

This script analyzes the camera tokens from the 3 experiments:
1. exp1_baseline: 8-frame standard pipeline
2. exp2_4frame: 4-frame [0,2,4,6] direct pass
3. exp3_extract4: 8-frame encode -> extract 4-frame features

Analysis includes:
- Cosine similarity between camera tokens
- MSE between camera tokens
- Token magnitude/norm analysis
- PCA visualization
- Per-dimension variance analysis
- Pose decoder sensitivity analysis
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict as AdictDict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.registries import MV_REGISTRY
from depth_anything_3.cfg import load_config
from depth_anything_3.utils.io.input_processor import InputProcessor


class CameraTokenAnalyzer:
    """
    Analyzer for camera tokens across different experiments.
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./analysis_output",
        debug: bool = False,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.debug = debug

        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self.dataset = MV_REGISTRY.get("eth3d")()
        self.input_processor = InputProcessor()
        self._model = None

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            print(f"[INFO] Loading model from {self.model_path}")
            self._model = DepthAnything3.from_pretrained(self.model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(device)
            self._model.eval()
        return self._model

    def _sample_frames(self, scene_data: AdictDict, num_frames: int = 8) -> AdictDict:
        """Sample frames deterministically."""
        total_frames = len(scene_data.image_files)
        if total_frames <= num_frames:
            return scene_data

        np.random.seed(42)
        indices = sorted(
            np.random.choice(total_frames, num_frames, replace=False).tolist()
        )

        sampled = AdictDict()
        sampled.image_files = [scene_data.image_files[i] for i in indices]
        sampled.extrinsics = scene_data.extrinsics[indices]
        sampled.intrinsics = scene_data.intrinsics[indices]
        sampled.aux = scene_data.aux
        return sampled

    def _preprocess_images(self, scene_data: AdictDict) -> torch.Tensor:
        """Load and preprocess images."""
        imgs_tensor, _, _ = self.input_processor(
            scene_data.image_files,
            scene_data.extrinsics,
            scene_data.intrinsics,
            process_res=504,
            process_res_method="upper_bound_resize",
            num_workers=4,
            print_progress=False,
        )
        device = self.model._get_model_device()
        imgs_tensor = imgs_tensor.to(device).float()
        if imgs_tensor.dim() == 4:
            imgs_tensor = imgs_tensor[None]
        return imgs_tensor

    def extract_camera_tokens(
        self, images: torch.Tensor, frame_indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract camera tokens for different experiment configurations.

        Returns dict with:
        - 'exp1_tokens': Camera tokens from 8-frame pass (all 8)
        - 'exp2_tokens': Camera tokens from 4-frame pass ([0,2,4,6])
        - 'exp3_tokens': Camera tokens extracted from 8-frame pass ([0,2,4,6])
        - 'exp1_feats': Full features from exp1
        - 'exp2_feats': Full features from exp2
        - 'exp3_feats': Full features from exp3
        """
        net = self.model.model if hasattr(self.model, 'model') else self.model
        device = images.device
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        indices_4frame = [0, 2, 4, 6]

        results = {}

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                # Exp1: 8-frame pass
                feats_8, _, H, W = net.forward_backbone_only(
                    images,
                    extrinsics=None,
                    intrinsics=None,
                    ref_view_strategy="first",
                )
                # feats_8 is list of (patch_feat, cam_token) tuples
                # cam_token shape: [B, 8, D]
                results['exp1_tokens'] = [f[1].float() for f in feats_8]  # All layers
                results['exp1_feats'] = feats_8

                # Exp2: 4-frame pass
                images_4 = images[:, indices_4frame]
                feats_4, _, _, _ = net.forward_backbone_only(
                    images_4,
                    extrinsics=None,
                    intrinsics=None,
                    ref_view_strategy="first",
                )
                results['exp2_tokens'] = [f[1].float() for f in feats_4]
                results['exp2_feats'] = feats_4

                # Exp3: Extract from 8-frame pass
                exp3_tokens = []
                exp3_feats = []
                for patch_feat, cam_token in feats_8:
                    exp3_tokens.append(cam_token[:, indices_4frame].float())
                    exp3_feats.append((
                        patch_feat[:, indices_4frame],
                        cam_token[:, indices_4frame]
                    ))
                results['exp3_tokens'] = exp3_tokens
                results['exp3_feats'] = exp3_feats

                # Also get pose decoder outputs
                results['exp1_pose'] = net.forward_head_only(feats_8, H, W, process_camera=True, process_sky=False)
                results['exp2_pose'] = net.forward_head_only(feats_4, H, W, process_camera=True, process_sky=False)
                results['exp3_pose'] = net.forward_head_only(exp3_feats, H, W, process_camera=True, process_sky=False)

        return results

    def compute_token_statistics(
        self,
        exp1_tokens: List[torch.Tensor],
        exp2_tokens: List[torch.Tensor],
        exp3_tokens: List[torch.Tensor],
    ) -> Dict:
        """
        Compute statistics comparing camera tokens across experiments.

        Compare exp2 vs exp3 (both have 4 frames [0,2,4,6])
        """
        stats = AdictDict()

        num_layers = len(exp1_tokens)

        for layer_idx in range(num_layers):
            layer_stats = AdictDict()

            # Get tokens for this layer
            t1 = exp1_tokens[layer_idx]  # [B, 8, D]
            t2 = exp2_tokens[layer_idx]  # [B, 4, D]
            t3 = exp3_tokens[layer_idx]  # [B, 4, D]

            # Extract corresponding frames from exp1 for comparison
            t1_subset = t1[:, [0, 2, 4, 6]]  # [B, 4, D]

            # Flatten for statistics
            t1_flat = t1_subset.reshape(-1, t1_subset.shape[-1])  # [B*4, D]
            t2_flat = t2.reshape(-1, t2.shape[-1])
            t3_flat = t3.reshape(-1, t3.shape[-1])

            # 1. Cosine similarity
            # exp2 vs exp3
            cos_sim_23 = F.cosine_similarity(t2_flat, t3_flat, dim=-1)
            layer_stats.cos_sim_exp2_exp3_mean = cos_sim_23.mean().item()
            layer_stats.cos_sim_exp2_exp3_std = cos_sim_23.std().item()
            layer_stats.cos_sim_exp2_exp3_min = cos_sim_23.min().item()

            # exp1_subset vs exp2
            cos_sim_12 = F.cosine_similarity(t1_flat, t2_flat, dim=-1)
            layer_stats.cos_sim_exp1_exp2_mean = cos_sim_12.mean().item()

            # exp1_subset vs exp3
            cos_sim_13 = F.cosine_similarity(t1_flat, t3_flat, dim=-1)
            layer_stats.cos_sim_exp1_exp3_mean = cos_sim_13.mean().item()

            # 2. MSE
            mse_23 = F.mse_loss(t2_flat, t3_flat, reduction='none').mean(dim=-1)
            layer_stats.mse_exp2_exp3_mean = mse_23.mean().item()
            layer_stats.mse_exp2_exp3_std = mse_23.std().item()
            layer_stats.mse_exp2_exp3_max = mse_23.max().item()

            mse_12 = F.mse_loss(t1_flat, t2_flat, reduction='none').mean(dim=-1)
            layer_stats.mse_exp1_exp2_mean = mse_12.mean().item()

            mse_13 = F.mse_loss(t1_flat, t3_flat, reduction='none').mean(dim=-1)
            layer_stats.mse_exp1_exp3_mean = mse_13.mean().item()

            # 3. L2 norm
            layer_stats.norm_exp1 = t1_flat.norm(dim=-1).mean().item()
            layer_stats.norm_exp2 = t2_flat.norm(dim=-1).mean().item()
            layer_stats.norm_exp3 = t3_flat.norm(dim=-1).mean().item()

            # 4. Per-dimension difference analysis
            diff_23 = (t2_flat - t3_flat).abs()  # [B*4, D]
            layer_stats.diff_23_mean_per_dim = diff_23.mean(dim=0).cpu().numpy()  # [D]
            layer_stats.diff_23_std_per_dim = diff_23.std(dim=0).cpu().numpy()
            layer_stats.diff_23_max_per_dim = diff_23.max(dim=0).values.cpu().numpy()

            # 5. Variance per dimension
            layer_stats.var_exp2_per_dim = t2_flat.var(dim=0).cpu().numpy()
            layer_stats.var_exp3_per_dim = t3_flat.var(dim=0).cpu().numpy()

            # 6. Token difference vector analysis
            diff_vector = (t2_flat - t3_flat).mean(dim=0)  # [D]
            layer_stats.diff_vector_norm = diff_vector.norm().item()
            layer_stats.diff_vector = diff_vector.cpu().numpy()

            stats[f'layer_{layer_idx}'] = layer_stats

        return stats

    def analyze_pose_sensitivity(
        self,
        results: Dict,
    ) -> Dict:
        """
        Analyze how small differences in camera tokens lead to large pose differences.
        """
        analysis = AdictDict()

        # Get poses
        pose1 = results['exp1_pose']
        pose2 = results['exp2_pose']
        pose3 = results['exp3_pose']

        # Compare extrinsics (translation and rotation)
        if 'extrinsics' in pose2 and 'extrinsics' in pose3:
            ext2 = pose2.extrinsics.float()  # [B, 4, 3, 4]
            ext3 = pose3.extrinsics.float()

            # Translation difference
            trans2 = ext2[..., :3, 3]  # [B, 4, 3]
            trans3 = ext3[..., :3, 3]
            trans_diff = (trans2 - trans3).norm(dim=-1)  # [B, 4]

            analysis.translation_diff_mean = trans_diff.mean().item()
            analysis.translation_diff_std = trans_diff.std().item()
            analysis.translation_diff_max = trans_diff.max().item()
            analysis.translation_diff_per_frame = trans_diff.mean(dim=0).cpu().numpy()

            # Rotation difference (Frobenius norm of rotation matrix difference)
            rot2 = ext2[..., :3, :3]  # [B, 4, 3, 3]
            rot3 = ext3[..., :3, :3]
            rot_diff = (rot2 - rot3).norm(dim=(-2, -1))  # [B, 4]

            analysis.rotation_diff_mean = rot_diff.mean().item()
            analysis.rotation_diff_std = rot_diff.std().item()
            analysis.rotation_diff_max = rot_diff.max().item()
            analysis.rotation_diff_per_frame = rot_diff.mean(dim=0).cpu().numpy()

        # Compare intrinsics
        if 'intrinsics' in pose2 and 'intrinsics' in pose3:
            int2 = pose2.intrinsics.float()  # [B, 4, 3, 3]
            int3 = pose3.intrinsics.float()

            # Focal length difference
            fx2, fy2 = int2[..., 0, 0], int2[..., 1, 1]
            fx3, fy3 = int3[..., 0, 0], int3[..., 1, 1]

            analysis.focal_diff_x_mean = (fx2 - fx3).abs().mean().item()
            analysis.focal_diff_y_mean = (fy2 - fy3).abs().mean().item()

        return analysis

    def visualize_tokens(
        self,
        exp1_tokens: List[torch.Tensor],
        exp2_tokens: List[torch.Tensor],
        exp3_tokens: List[torch.Tensor],
        scene_name: str,
    ):
        """Create visualizations for camera tokens."""

        # Use last layer tokens (most processed)
        t1 = exp1_tokens[-1][:, [0, 2, 4, 6]].cpu().numpy()  # [B, 4, D]
        t2 = exp2_tokens[-1].cpu().numpy()
        t3 = exp3_tokens[-1].cpu().numpy()

        B, num_frames, D = t2.shape

        # Flatten for visualization
        t1_flat = t1.reshape(-1, D)
        t2_flat = t2.reshape(-1, D)
        t3_flat = t3.reshape(-1, D)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. PCA visualization
        all_tokens = np.vstack([t1_flat, t2_flat, t3_flat])
        pca = PCA(n_components=2)
        tokens_pca = pca.fit_transform(all_tokens)

        n = len(t1_flat)
        ax = axes[0, 0]
        ax.scatter(tokens_pca[:n, 0], tokens_pca[:n, 1], c='blue', label='Exp1 (8fr subset)', alpha=0.7, s=100)
        ax.scatter(tokens_pca[n:2*n, 0], tokens_pca[n:2*n, 1], c='green', label='Exp2 (4fr direct)', alpha=0.7, s=100)
        ax.scatter(tokens_pca[2*n:, 0], tokens_pca[2*n:, 1], c='red', label='Exp3 (8fr extract)', alpha=0.7, s=100)
        ax.set_title(f'PCA of Camera Tokens (Last Layer)\n{scene_name}')
        ax.legend()
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

        # 2. Token difference heatmap (exp2 vs exp3)
        ax = axes[0, 1]
        diff = np.abs(t2_flat - t3_flat)
        im = ax.imshow(diff, aspect='auto', cmap='hot')
        ax.set_title('Absolute Difference: Exp2 vs Exp3')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Frame')
        plt.colorbar(im, ax=ax)

        # 3. Cosine similarity per frame
        ax = axes[0, 2]
        cos_sims = []
        for i in range(num_frames):
            cos_sim = F.cosine_similarity(
                torch.from_numpy(t2[0, i:i+1]),
                torch.from_numpy(t3[0, i:i+1]),
                dim=-1
            ).item()
            cos_sims.append(cos_sim)

        frame_labels = ['Frame 0', 'Frame 2', 'Frame 4', 'Frame 6']
        bars = ax.bar(frame_labels, cos_sims, color=['C0', 'C1', 'C2', 'C3'])
        ax.set_ylim(0.99, 1.001)
        ax.set_title('Cosine Similarity per Frame (Exp2 vs Exp3)')
        ax.set_ylabel('Cosine Similarity')
        for bar, sim in zip(bars, cos_sims):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{sim:.6f}', ha='center', va='bottom', fontsize=9)

        # 4. Per-dimension variance comparison
        ax = axes[1, 0]
        var2 = np.var(t2_flat, axis=0)
        var3 = np.var(t3_flat, axis=0)

        # Show top 50 dimensions with highest variance difference
        var_diff = np.abs(var2 - var3)
        top_dims = np.argsort(var_diff)[-50:]

        x = np.arange(50)
        width = 0.35
        ax.bar(x - width/2, var2[top_dims], width, label='Exp2', alpha=0.7)
        ax.bar(x + width/2, var3[top_dims], width, label='Exp3', alpha=0.7)
        ax.set_title('Variance Comparison (Top 50 dims by diff)')
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Variance')
        ax.legend()

        # 5. Difference magnitude per dimension
        ax = axes[1, 1]
        mean_diff = np.mean(np.abs(t2_flat - t3_flat), axis=0)
        top_diff_dims = np.argsort(mean_diff)[-100:]
        ax.bar(range(100), mean_diff[top_diff_dims])
        ax.set_title('Mean Abs Difference (Top 100 dims)')
        ax.set_xlabel('Dimension Rank')
        ax.set_ylabel('Mean |Exp2 - Exp3|')

        # 6. Token norm comparison
        ax = axes[1, 2]
        norms = {
            'Exp1 (subset)': np.linalg.norm(t1_flat, axis=1),
            'Exp2 (4fr)': np.linalg.norm(t2_flat, axis=1),
            'Exp3 (extract)': np.linalg.norm(t3_flat, axis=1),
        }
        ax.boxplot([norms['Exp1 (subset)'], norms['Exp2 (4fr)'], norms['Exp3 (extract)']],
                   labels=['Exp1', 'Exp2', 'Exp3'])
        ax.set_title('Token L2 Norm Distribution')
        ax.set_ylabel('L2 Norm')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'camera_token_analysis_{scene_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved visualization to {self.output_dir}/camera_token_analysis_{scene_name}.png")

    def visualize_pose_decoder_analysis(
        self,
        results: Dict,
        stats: Dict,
        scene_name: str,
    ):
        """Visualize pose decoder sensitivity analysis."""

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Get tokens from last layer
        t2 = results['exp2_tokens'][-1].cpu().numpy()
        t3 = results['exp3_tokens'][-1].cpu().numpy()

        # 1. Token difference vs Pose difference scatter
        ax = axes[0, 0]

        # Per-frame token MSE
        token_mse_per_frame = np.mean((t2 - t3) ** 2, axis=-1).flatten()

        # Per-frame pose difference
        pose_analysis = self.analyze_pose_sensitivity(results)
        trans_diff = pose_analysis.get('translation_diff_per_frame', np.zeros(4))
        rot_diff = pose_analysis.get('rotation_diff_per_frame', np.zeros(4))

        frame_labels = ['Fr0', 'Fr2', 'Fr4', 'Fr6']
        colors = ['C0', 'C1', 'C2', 'C3']

        for i, (label, color) in enumerate(zip(frame_labels, colors)):
            ax.scatter(token_mse_per_frame[i], trans_diff[i],
                      c=color, s=150, label=f'{label} (trans)', marker='o')
            ax.scatter(token_mse_per_frame[i], rot_diff[i],
                      c=color, s=150, label=f'{label} (rot)', marker='^')

        ax.set_xlabel('Token MSE')
        ax.set_ylabel('Pose Difference')
        ax.set_title('Token MSE vs Pose Difference per Frame')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. Amplification factor visualization
        ax = axes[0, 1]

        # Calculate amplification: pose_diff / token_diff
        token_diff_norm = np.linalg.norm(t2 - t3, axis=-1).flatten()

        if np.any(token_diff_norm > 1e-8):
            trans_amplification = trans_diff / (token_diff_norm + 1e-8)
            rot_amplification = rot_diff / (token_diff_norm + 1e-8)

            x = np.arange(4)
            width = 0.35
            ax.bar(x - width/2, trans_amplification, width, label='Translation Amp', alpha=0.7)
            ax.bar(x + width/2, rot_amplification, width, label='Rotation Amp', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(frame_labels)
            ax.set_ylabel('Amplification Factor')
            ax.set_title('Pose Decoder Amplification\n(Pose Diff / Token Diff)')
            ax.legend()

        # 3. Layer-wise token difference
        ax = axes[1, 0]

        layer_mse = []
        layer_cos = []
        for layer_key in sorted([k for k in stats.keys() if k.startswith('layer_')]):
            layer_stats = stats[layer_key]
            layer_mse.append(layer_stats.mse_exp2_exp3_mean)
            layer_cos.append(1 - layer_stats.cos_sim_exp2_exp3_mean)  # 1 - cos for visibility

        x = range(len(layer_mse))
        ax.plot(x, layer_mse, 'o-', label='MSE', color='blue')
        ax2 = ax.twinx()
        ax2.plot(x, layer_cos, 's-', label='1 - CosSim', color='red')

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('MSE', color='blue')
        ax2.set_ylabel('1 - Cosine Similarity', color='red')
        ax.set_title('Token Difference Across Layers')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # 4. Dimension importance for pose (gradient-like analysis)
        ax = axes[1, 1]

        # Use the difference vector to identify important dimensions
        last_layer_stats = stats[f'layer_{len(stats)-1}']
        diff_per_dim = last_layer_stats.diff_23_mean_per_dim

        # Top 30 dimensions with highest difference
        top_dims = np.argsort(diff_per_dim)[-30:]
        ax.barh(range(30), diff_per_dim[top_dims])
        ax.set_yticks(range(30))
        ax.set_yticklabels([f'Dim {d}' for d in top_dims], fontsize=8)
        ax.set_xlabel('Mean Absolute Difference')
        ax.set_title('Top 30 Dimensions by Exp2-Exp3 Difference')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'pose_sensitivity_{scene_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved pose sensitivity analysis to {self.output_dir}/pose_sensitivity_{scene_name}.png")

    def run_analysis(self, scenes: Optional[List[str]] = None):
        """Run full analysis on specified scenes."""

        all_scenes = self.dataset.SCENES
        if scenes:
            all_scenes = [s for s in all_scenes if s in scenes]

        all_stats = []
        all_pose_analysis = []

        for scene in tqdm(all_scenes, desc="Analyzing scenes"):
            print(f"\n[ANALYSIS] Processing {scene}")

            # Get scene data
            scene_data = self.dataset.get_data(scene)
            sampled_data = self._sample_frames(scene_data, num_frames=8)

            # Check if we have enough frames
            if len(sampled_data.image_files) < 8:
                print(f"  [SKIP] {scene}: not enough frames")
                continue

            # Preprocess images
            images = self._preprocess_images(sampled_data)

            # Extract tokens
            results = self.extract_camera_tokens(images)

            # Compute statistics
            stats = self.compute_token_statistics(
                results['exp1_tokens'],
                results['exp2_tokens'],
                results['exp3_tokens'],
            )

            # Analyze pose sensitivity
            pose_analysis = self.analyze_pose_sensitivity(results)

            # Print summary
            last_layer = f'layer_{len(stats)-1}'
            print(f"  Token Stats (Last Layer):")
            print(f"    Cosine Sim (Exp2 vs Exp3): {stats[last_layer].cos_sim_exp2_exp3_mean:.6f}")
            print(f"    MSE (Exp2 vs Exp3): {stats[last_layer].mse_exp2_exp3_mean:.6f}")
            print(f"  Pose Difference:")
            print(f"    Translation diff: {pose_analysis.get('translation_diff_mean', 0):.6f}")
            print(f"    Rotation diff: {pose_analysis.get('rotation_diff_mean', 0):.6f}")

            # Visualizations
            self.visualize_tokens(
                results['exp1_tokens'],
                results['exp2_tokens'],
                results['exp3_tokens'],
                scene,
            )

            self.visualize_pose_decoder_analysis(results, stats, scene)

            all_stats.append((scene, stats, pose_analysis))

        # Aggregate statistics
        self.print_aggregate_stats(all_stats)

        return all_stats

    def print_aggregate_stats(self, all_stats: List):
        """Print aggregated statistics across all scenes."""

        print("\n" + "=" * 80)
        print("AGGREGATE STATISTICS ACROSS ALL SCENES")
        print("=" * 80)

        # Collect metrics
        cos_sims = []
        mses = []
        trans_diffs = []
        rot_diffs = []

        for scene, stats, pose_analysis in all_stats:
            last_layer = f'layer_{len(stats)-1}'
            cos_sims.append(stats[last_layer].cos_sim_exp2_exp3_mean)
            mses.append(stats[last_layer].mse_exp2_exp3_mean)
            trans_diffs.append(pose_analysis.get('translation_diff_mean', 0))
            rot_diffs.append(pose_analysis.get('rotation_diff_mean', 0))

        print(f"\nCamera Token Statistics (Exp2 vs Exp3, Last Layer):")
        print(f"  Cosine Similarity: {np.mean(cos_sims):.6f} ± {np.std(cos_sims):.6f}")
        print(f"  MSE:               {np.mean(mses):.6f} ± {np.std(mses):.6f}")

        print(f"\nPose Difference (Exp2 vs Exp3):")
        print(f"  Translation:       {np.mean(trans_diffs):.6f} ± {np.std(trans_diffs):.6f}")
        print(f"  Rotation:          {np.mean(rot_diffs):.6f} ± {np.std(rot_diffs):.6f}")

        # Amplification factor
        if np.mean(mses) > 1e-8:
            trans_amp = np.mean(trans_diffs) / np.sqrt(np.mean(mses))
            rot_amp = np.mean(rot_diffs) / np.sqrt(np.mean(mses))
            print(f"\nAmplification Factor (Pose Diff / sqrt(Token MSE)):")
            print(f"  Translation:       {trans_amp:.2f}x")
            print(f"  Rotation:          {rot_amp:.2f}x")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze camera tokens across experiments")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--scenes", type=str, nargs="+", help="Specific scenes to analyze")
    parser.add_argument("--output_dir", type=str, default="./analysis_output",
                       help="Output directory for visualizations")
    args, remaining = parser.parse_known_args()

    # Load config
    config_path = args.config or os.path.join(
        os.path.dirname(__file__), "eth3d_frame_experiment.yaml"
    )
    config = load_config(config_path, argv=remaining)

    analyzer = CameraTokenAnalyzer(
        model_path=config.model.path,
        output_dir=args.output_dir,
        debug=True,
    )

    analyzer.run_analysis(scenes=args.scenes)


if __name__ == "__main__":
    main()
