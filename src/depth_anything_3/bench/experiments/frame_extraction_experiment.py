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
Frame extraction experiments for ETH3D multiview benchmark.

Implements 4 experiments:
1. Exp1_8FrameBaseline: Standard 8-frame pipeline
2. Exp2_4FrameDirect: 4-frame [0,2,4,6] direct pass
3. Exp3_8FrameExtract4: 8-frame encode, extract 4-frame features, predict
4. Exp4_SpikeDimAblation: Ablation study on spike dimensions for pose
"""

from typing import Dict, List, Optional, Tuple

import torch
from addict import Dict as AdictDict

from depth_anything_3.bench.experiments.base_experiment import BaseFrameExperiment


class Exp1_8FrameBaseline(BaseFrameExperiment):
    """
    Experiment 1: 8-Frame Baseline

    Uses the current 8-frame selection (all frames 0-7).
    Runs standard benchmark with pose + recon_unposed.
    This serves as the baseline for comparison.
    """

    @property
    def experiment_name(self) -> str:
        return "exp1_8frame_baseline"

    def run_inference(
        self,
        images: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> AdictDict:
        """
        Run standard 8-frame inference through complete pipeline.

        Args:
            images: Input images [B, 8, 3, H, W]
            extrinsics: Optional GT extrinsics [B, 8, 4, 4]
            intrinsics: Optional GT intrinsics [B, 8, 3, 3]

        Returns:
            Model output with depth, extrinsics, intrinsics for all 8 frames
        """
        B, S, C, H, W = images.shape
        assert S == 8, f"Expected 8 frames, got {S}"

        # Standard forward pass through complete pipeline
        autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        with torch.no_grad():
            with torch.autocast(device_type=images.device.type, dtype=autocast_dtype):
                output = self.net(
                    images,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    ref_view_strategy=self.ref_view_strategy,
                )

        return output


class Exp2_4FrameDirect(BaseFrameExperiment):
    """
    Experiment 2: 4-Frame Direct Pass

    Selects frames [0,2,4,6] from input (4 frames).
    Passes these 4 frames through complete pipeline (encoder + prediction heads).
    Uses frame 0 as reference (consistent with "first" strategy).
    """

    FRAME_INDICES = [0, 2, 4, 6]

    @property
    def experiment_name(self) -> str:
        return "exp2_4frame_direct"

    def get_output_frames(self) -> List[int]:
        """Return the frame indices this experiment outputs predictions for."""
        return self.FRAME_INDICES

    def run_inference(
        self,
        images: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> AdictDict:
        """
        Run 4-frame inference by selecting frames [0,2,4,6].

        Args:
            images: Input images [B, 8, 3, H, W]
            extrinsics: Optional GT extrinsics [B, 8, 4, 4]
            intrinsics: Optional GT intrinsics [B, 8, 3, 3]

        Returns:
            Model output with depth, extrinsics, intrinsics for 4 frames
        """
        B, S, C, H, W = images.shape
        assert S == 8, f"Expected 8 frames, got {S}"

        # Select 4 frames: [0, 2, 4, 6]
        images_4 = images[:, self.FRAME_INDICES]  # [B, 4, 3, H, W]

        # Select corresponding GT if provided
        ext_4 = extrinsics[:, self.FRAME_INDICES] if extrinsics is not None else None
        int_4 = intrinsics[:, self.FRAME_INDICES] if intrinsics is not None else None

        # Standard forward pass with 4 frames
        autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        with torch.no_grad():
            with torch.autocast(device_type=images.device.type, dtype=autocast_dtype):
                output = self.net(
                    images_4,
                    extrinsics=ext_4,
                    intrinsics=int_4,
                    ref_view_strategy=self.ref_view_strategy,
                )

        return output


class Exp3_8FrameExtract4(BaseFrameExperiment):
    """
    Experiment 3: 8-Frame Encode, Extract 4-Frame Features, Predict

    This is the key experiment:
    1. Pass all 8 frames through encoder (cross-frame attention occurs)
    2. After encoding, extract features for frames [0,2,4,6] only
    3. Extract both patch features AND camera tokens for these 4 frames
    4. Pass extracted 4-frame features to prediction heads

    The hypothesis is that cross-frame attention on 8 frames provides
    richer features that benefit prediction even when only using 4 frames.
    """

    FRAME_INDICES = [0, 2, 4, 6]

    @property
    def experiment_name(self) -> str:
        return "exp3_8frame_extract4"

    def get_output_frames(self) -> List[int]:
        """Return the frame indices this experiment outputs predictions for."""
        return self.FRAME_INDICES

    def _extract_frame_features(
        self,
        feats: List[Tuple[torch.Tensor, torch.Tensor]],
        indices: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract features for specific frame indices.

        Args:
            feats: List of (patch_feat, cam_token) tuples from backbone
                   - patch_feat shape: [B, S, N, C] where N = num_patches + 1
                   - cam_token shape: [B, S, D]
            indices: Frame indices to extract

        Returns:
            List of (extracted_patch_feat, extracted_cam_token) tuples
            - extracted_patch_feat shape: [B, len(indices), N, C]
            - extracted_cam_token shape: [B, len(indices), D]
        """
        extracted_feats = []

        for patch_feat, cam_token in feats:
            # Extract patch features for selected frames
            # patch_feat: [B, S, N, C] -> [B, 4, N, C]
            extracted_patch = patch_feat[:, indices]

            # Extract camera tokens for selected frames
            # cam_token: [B, S, D] -> [B, 4, D]
            extracted_cam = cam_token[:, indices]

            extracted_feats.append((extracted_patch, extracted_cam))

        return extracted_feats

    def run_inference(
        self,
        images: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> AdictDict:
        """
        Run 8-frame backbone, extract 4-frame features, then predict.

        Pipeline:
        1. forward_backbone_only(8 frames) -> features with cross-frame attention
        2. Extract features for frames [0,2,4,6]
        3. forward_head_only(4-frame features) -> predictions

        Args:
            images: Input images [B, 8, 3, H, W]
            extrinsics: Optional GT extrinsics (not used in unposed mode)
            intrinsics: Optional GT intrinsics (not used in unposed mode)

        Returns:
            Model output with depth, extrinsics, intrinsics for 4 frames
        """
        B, S, C, H, W = images.shape
        assert S == 8, f"Expected 8 frames, got {S}"

        device = images.device
        autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                # Step 1: Run backbone on all 8 frames
                # This allows cross-frame attention to occur
                feats, aux_feats, H_out, W_out = self.net.forward_backbone_only(
                    images,
                    extrinsics=None,  # Unposed mode
                    intrinsics=None,
                    ref_view_strategy=self.ref_view_strategy,
                )

                # Step 2: Extract features for frames [0, 2, 4, 6]
                # feats is a list of (patch_features, camera_tokens) tuples
                # Each layer has:
                #   - patch_features: [B, 8, N, C]
                #   - camera_tokens: [B, 8, D]
                extracted_feats = self._extract_frame_features(feats, self.FRAME_INDICES)

                # Step 3: Run prediction head with extracted 4-frame features
                output = self.net.forward_head_only(
                    extracted_feats,
                    H_out,
                    W_out,
                    process_camera=True,
                    process_sky=True,
                )

        return output


class Exp4_SpikeDimAblation(BaseFrameExperiment):
    """
    Experiment 4: Spike Dimension Ablation Study

    Tests whether spike dimensions (high-MSE dims between exp2/exp3 camera tokens)
    are responsible for pose estimation differences.

    The experiment:
    1. Runs both 8-frame and 4-frame backbones to get exp2 and exp3 camera tokens
    2. Identifies spike dimensions (top 10% by MSE)
    3. Creates 4 ablation variants:
       - baseline: exp2 tokens unchanged
       - spike_fixed: exp2 with spike dims replaced by exp3
       - nonspike_fixed: exp2 with non-spike dims replaced by exp3 (control)
       - teacher: exp3 tokens (upper bound)
    4. Runs head on each variant and returns all outputs

    This answers: "Can perfectly aligning spike dims close the pose gap?"
    """

    FRAME_INDICES = [0, 2, 4, 6]
    SPIKE_PERCENTILE = 97  # Top 3% MSE dims are "spike dims"
    ABLATION_CONDITIONS = ["baseline", "spike_fixed", "nonspike_fixed", "first_half_fixed", "last_half_fixed", "student_copy_first_to_last", "student_copy_last_to_first", "teacher"]

    @property
    def experiment_name(self) -> str:
        return "exp4_spike_ablation"

    def get_output_frames(self) -> List[int]:
        """Return the frame indices this experiment outputs predictions for."""
        return self.FRAME_INDICES

    def _extract_frame_features(
        self,
        feats: List[Tuple[torch.Tensor, torch.Tensor]],
        indices: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract features for specific frame indices."""
        extracted_feats = []
        for patch_feat, cam_token in feats:
            extracted_patch = patch_feat[:, indices]
            extracted_cam = cam_token[:, indices]
            extracted_feats.append((extracted_patch, extracted_cam))
        return extracted_feats

    def _identify_spike_dims(
        self,
        student_tokens: List[torch.Tensor],  # List of [B, 4, D] from exp2
        teacher_tokens: List[torch.Tensor],  # List of [B, 4, D] from exp3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find dimensions with highest MSE between student (exp2) and teacher (exp3).

        Uses the last layer's camera tokens for spike identification.

        Returns:
            Tuple of (spike_dims, non_spike_dims, per_dim_mse)
        """
        # Use last layer tokens
        student = student_tokens[-1]  # [B, 4, D]
        teacher = teacher_tokens[-1]  # [B, 4, D]

        # Compute per-dimension MSE across batch and frames
        per_dim_mse = ((student - teacher) ** 2).mean(dim=[0, 1])  # [D]

        # Find threshold for top SPIKE_PERCENTILE%
        threshold = torch.quantile(per_dim_mse, self.SPIKE_PERCENTILE / 100)

        # Identify spike and non-spike dimensions
        spike_dims = torch.where(per_dim_mse > threshold)[0]
        non_spike_dims = torch.where(per_dim_mse <= threshold)[0]

        return spike_dims, non_spike_dims, per_dim_mse

    def _create_ablation_features(
        self,
        exp2_feats: List[Tuple[torch.Tensor, torch.Tensor]],
        exp3_feats: List[Tuple[torch.Tensor, torch.Tensor]],
        spike_dims: torch.Tensor,
        non_spike_dims: torch.Tensor,
    ) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Create ablation variants by manipulating camera tokens.

        For each variant, we modify only the camera tokens while keeping
        patch features from exp2 (baseline).

        Returns:
            Dict mapping condition name to feature list
        """
        ablation_feats = {}

        # Get dimension size from first feature
        dim_size = exp2_feats[0][1].shape[-1]  # D dimension
        half_dim = dim_size // 2

        # 1. Baseline: exp2 features unchanged
        ablation_feats["baseline"] = exp2_feats

        # 2. Spike-fixed: Replace spike dims in exp2 cam tokens with exp3 values
        spike_fixed_feats = []
        for (patch_2, cam_2), (patch_3, cam_3) in zip(exp2_feats, exp3_feats):
            # Clone exp2 camera tokens
            cam_fixed = cam_2.clone()
            # Replace spike dimensions with exp3 values
            cam_fixed[..., spike_dims] = cam_3[..., spike_dims]
            # Keep patch features from exp2
            spike_fixed_feats.append((patch_2, cam_fixed))
        ablation_feats["spike_fixed"] = spike_fixed_feats

        # 3. Non-spike-fixed: Replace non-spike dims (control experiment)
        nonspike_fixed_feats = []
        for (patch_2, cam_2), (patch_3, cam_3) in zip(exp2_feats, exp3_feats):
            cam_fixed = cam_2.clone()
            cam_fixed[..., non_spike_dims] = cam_3[..., non_spike_dims]
            nonspike_fixed_feats.append((patch_2, cam_fixed))
        ablation_feats["nonspike_fixed"] = nonspike_fixed_feats

        # 4. First-half-fixed: Replace first half of dims [0:768] with exp3 values
        first_half_fixed_feats = []
        for (patch_2, cam_2), (patch_3, cam_3) in zip(exp2_feats, exp3_feats):
            cam_fixed = cam_2.clone()
            cam_fixed[..., :half_dim] = cam_3[..., :half_dim]
            first_half_fixed_feats.append((patch_2, cam_fixed))
        ablation_feats["first_half_fixed"] = first_half_fixed_feats

        # 5. Last-half-fixed: Replace last half of dims [768:1536] with exp3 values
        last_half_fixed_feats = []
        for (patch_2, cam_2), (patch_3, cam_3) in zip(exp2_feats, exp3_feats):
            cam_fixed = cam_2.clone()
            cam_fixed[..., half_dim:] = cam_3[..., half_dim:]
            last_half_fixed_feats.append((patch_2, cam_fixed))
        ablation_feats["last_half_fixed"] = last_half_fixed_feats

        # 6. Student copy first to last: Copy student's first half [0:768] to last half [768:1536]
        student_copy_first_to_last_feats = []
        for (patch_2, cam_2), _ in zip(exp2_feats, exp3_feats):
            cam_fixed = cam_2.clone()
            cam_fixed[..., half_dim:] = cam_2[..., :half_dim]  # Copy first half to last half
            student_copy_first_to_last_feats.append((patch_2, cam_fixed))
        ablation_feats["student_copy_first_to_last"] = student_copy_first_to_last_feats

        # 7. Student copy last to first: Copy student's last half [768:1536] to first half [0:768]
        student_copy_last_to_first_feats = []
        for (patch_2, cam_2), _ in zip(exp2_feats, exp3_feats):
            cam_fixed = cam_2.clone()
            cam_fixed[..., :half_dim] = cam_2[..., half_dim:]  # Copy last half to first half
            student_copy_last_to_first_feats.append((patch_2, cam_fixed))
        ablation_feats["student_copy_last_to_first"] = student_copy_last_to_first_feats

        # 8. Teacher: exp3 features (upper bound)
        ablation_feats["teacher"] = exp3_feats

        return ablation_feats

    def run_inference(
        self,
        images: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> AdictDict:
        """
        Run spike dimension ablation experiment.

        Pipeline:
        1. Run 8-frame backbone -> exp3-style features
        2. Run 4-frame backbone -> exp2-style features
        3. Identify spike dimensions from camera tokens
        4. Create ablation variants
        5. Run head on each variant

        Args:
            images: Input images [B, 8, 3, H, W]
            extrinsics: Optional GT extrinsics (not used)
            intrinsics: Optional GT intrinsics (not used)

        Returns:
            AdictDict with outputs for each ablation condition:
            - outputs["baseline"]: Output from exp2 tokens
            - outputs["spike_fixed"]: Output with spike dims from exp3
            - outputs["nonspike_fixed"]: Output with non-spike dims from exp3
            - outputs["teacher"]: Output from exp3 tokens
            - spike_info: Dict with spike dimension metadata
        """
        B, S, C, H, W = images.shape
        assert S == 8, f"Expected 8 frames, got {S}"

        device = images.device
        autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                # Step 1: Run 8-frame backbone (for exp3-style features)
                feats_8, aux_feats, H_out, W_out = self.net.forward_backbone_only(
                    images,
                    extrinsics=None,
                    intrinsics=None,
                    ref_view_strategy=self.ref_view_strategy,
                )

                # Extract 4-frame features from 8-frame pass (exp3)
                exp3_feats = self._extract_frame_features(feats_8, self.FRAME_INDICES)

                # Step 2: Run 4-frame backbone (for exp2-style features)
                images_4 = images[:, self.FRAME_INDICES]  # [B, 4, 3, H, W]
                feats_4, _, _, _ = self.net.forward_backbone_only(
                    images_4,
                    extrinsics=None,
                    intrinsics=None,
                    ref_view_strategy=self.ref_view_strategy,
                )
                exp2_feats = feats_4  # Already 4 frames

                # Step 3: Identify spike dimensions
                exp2_cam_tokens = [f[1] for f in exp2_feats]
                exp3_cam_tokens = [f[1] for f in exp3_feats]
                spike_dims, non_spike_dims, per_dim_mse = self._identify_spike_dims(
                    exp2_cam_tokens, exp3_cam_tokens
                )

                # Step 4: Create ablation variants
                ablation_feats = self._create_ablation_features(
                    exp2_feats, exp3_feats, spike_dims, non_spike_dims
                )

                # Step 5: Run head on each variant
                outputs = AdictDict()
                for condition in self.ABLATION_CONDITIONS:
                    condition_output = self.net.forward_head_only(
                        ablation_feats[condition],
                        H_out,
                        W_out,
                        process_camera=True,
                        process_sky=True,
                    )
                    outputs[condition] = condition_output

                # Add spike dimension metadata
                outputs.spike_info = AdictDict(
                    spike_dims=spike_dims.cpu(),
                    non_spike_dims=non_spike_dims.cpu(),
                    per_dim_mse=per_dim_mse.cpu(),
                    num_spike_dims=len(spike_dims),
                    num_non_spike_dims=len(non_spike_dims),
                    spike_percentile=self.SPIKE_PERCENTILE,
                )

        return outputs


class Exp5_TokenReplacement(BaseFrameExperiment):
    """
    Experiment 5: Local Token Replacement Analysis

    Tests whether high-MSE local tokens are outliers hurting performance
    or important features by:
    1. Run teacher with 8 views, extract local tokens for frames [0,2,4,6]
    2. Run student with 4 views (frames [0,2,4,6])
    3. Compute per-token MSE between local features
    4. Replace top X% high-MSE or bottom (100-X)% low-MSE tokens
    5. Decode and compare results
    """

    FRAME_INDICES = [0, 2, 4, 6]
    EMBED_DIM = 1536  # Giant model embedding dimension
    REPLACEMENT_PERCENTILES = [10, 20, 30, 50]

    @property
    def experiment_name(self) -> str:
        return "exp5_token_replacement"

    def get_output_frames(self) -> List[int]:
        return self.FRAME_INDICES

    def _extract_frame_features(
        self,
        feats: List[Tuple[torch.Tensor, torch.Tensor]],
        indices: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract features for specific frame indices."""
        extracted_feats = []
        for patch_feat, cam_token in feats:
            extracted_patch = patch_feat[:, indices]
            extracted_cam = cam_token[:, indices]
            extracted_feats.append((extracted_patch, extracted_cam))
        return extracted_feats

    def _split_local_global(
        self, feats: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split last layer features into local and global components."""
        # Last layer features: [B, S, N, 3072] -> local [B, S, N, 1536], global [B, S, N, 1536]
        full_feats = feats[-1][0]  # [B, S, N, 3072]
        local_feats = full_feats[..., :self.EMBED_DIM]
        global_feats = full_feats[..., self.EMBED_DIM:]
        return local_feats, global_feats

    def _compute_token_mse(
        self,
        teacher_local: torch.Tensor,
        student_local: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token MSE between teacher and student local features."""
        # MSE per token (mean over feature dim)
        mse_per_token = ((teacher_local - student_local) ** 2).mean(dim=-1)  # [B, S, N]
        return mse_per_token

    def _replace_tokens_by_percentile(
        self,
        teacher_local: torch.Tensor,
        student_local: torch.Tensor,
        mse_per_token: torch.Tensor,
        percentile: float,
        replace_high: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replace student local tokens based on MSE percentile."""
        mse_flat = mse_per_token.reshape(-1)

        if replace_high:
            threshold = torch.quantile(mse_flat, 1 - percentile / 100)
            mask = mse_per_token >= threshold
        else:
            threshold = torch.quantile(mse_flat, percentile / 100)
            mask = mse_per_token <= threshold

        modified_local = student_local.clone()
        modified_local[mask] = teacher_local[mask]

        return modified_local, mask

    def _create_modified_feats(
        self,
        student_feats: List[Tuple[torch.Tensor, torch.Tensor]],
        modified_local: torch.Tensor,
        student_global: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create modified feats list with replaced local tokens."""
        modified_full = torch.cat([modified_local, student_global], dim=-1)

        modified_feats = []
        for i, (feat, cam_token) in enumerate(student_feats):
            if i == len(student_feats) - 1:
                modified_feats.append((modified_full, cam_token))
            else:
                modified_feats.append((feat, cam_token))

        return modified_feats

    def run_inference(
        self,
        images: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> AdictDict:
        """Run token replacement experiment."""
        B, S, C, H, W = images.shape
        assert S == 8, f"Expected 8 frames, got {S}"

        device = images.device
        autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                # Step 1: Run 8-frame backbone (teacher)
                feats_8, _, H_out, W_out = self.net.forward_backbone_only(
                    images, ref_view_strategy=self.ref_view_strategy
                )
                teacher_feats = self._extract_frame_features(feats_8, self.FRAME_INDICES)
                teacher_local, teacher_global = self._split_local_global(teacher_feats)

                # Step 2: Run 4-frame backbone (student)
                images_4 = images[:, self.FRAME_INDICES]
                student_feats, _, _, _ = self.net.forward_backbone_only(
                    images_4, ref_view_strategy=self.ref_view_strategy
                )
                student_local, student_global = self._split_local_global(student_feats)

                # Step 3: Compute per-token MSE
                mse_per_token = self._compute_token_mse(teacher_local, student_local)

                # Step 4: Create outputs for each condition
                outputs = AdictDict()

                # Baseline: student features unchanged
                baseline_output = self.net.forward_head_only(
                    student_feats, H_out, W_out, process_camera=True, process_sky=True
                )
                outputs["baseline"] = baseline_output

                # Teacher: teacher features (upper bound)
                teacher_output = self.net.forward_head_only(
                    teacher_feats, H_out, W_out, process_camera=True, process_sky=True
                )
                outputs["teacher"] = teacher_output

                # Replacement experiments
                for pct in self.REPLACEMENT_PERCENTILES:
                    # Replace HIGH MSE tokens
                    mod_local_high, mask_high = self._replace_tokens_by_percentile(
                        teacher_local, student_local, mse_per_token, pct, replace_high=True
                    )
                    mod_feats_high = self._create_modified_feats(
                        student_feats, mod_local_high, student_global
                    )
                    out_high = self.net.forward_head_only(
                        mod_feats_high, H_out, W_out, process_camera=True, process_sky=True
                    )
                    outputs[f"replace_high_{pct}pct"] = out_high

                    # Replace LOW MSE tokens (complement)
                    mod_local_low, mask_low = self._replace_tokens_by_percentile(
                        teacher_local, student_local, mse_per_token, 100 - pct, replace_high=False
                    )
                    mod_feats_low = self._create_modified_feats(
                        student_feats, mod_local_low, student_global
                    )
                    out_low = self.net.forward_head_only(
                        mod_feats_low, H_out, W_out, process_camera=True, process_sky=True
                    )
                    outputs[f"replace_low_{100-pct}pct"] = out_low

                # Store MSE statistics
                outputs.mse_info = AdictDict(
                    mean=mse_per_token.mean().item(),
                    std=mse_per_token.std().item(),
                    min=mse_per_token.min().item(),
                    max=mse_per_token.max().item(),
                    median=mse_per_token.median().item(),
                )

        return outputs


def create_experiment(
    experiment_type: str,
    model,
    **kwargs,
) -> BaseFrameExperiment:
    """
    Factory function to create experiment instances.

    Args:
        experiment_type: One of "exp1_baseline", "exp2_4frame", "exp3_extract4",
                        "exp4_spike_ablation"
        model: DepthAnything3 API instance
        **kwargs: Additional arguments passed to experiment constructor

    Returns:
        Experiment instance
    """
    experiments = {
        "exp1_baseline": Exp1_8FrameBaseline,
        "exp2_4frame": Exp2_4FrameDirect,
        "exp3_extract4": Exp3_8FrameExtract4,
        "exp4_spike_ablation": Exp4_SpikeDimAblation,
        "exp5_token_replacement": Exp5_TokenReplacement,
    }

    if experiment_type not in experiments:
        raise ValueError(
            f"Unknown experiment: {experiment_type}. "
            f"Available: {list(experiments.keys())}"
        )

    return experiments[experiment_type](model, **kwargs)
