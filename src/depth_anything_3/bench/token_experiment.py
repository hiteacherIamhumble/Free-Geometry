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
Token manipulation utilities for frame sampling experiments.

This module provides utilities for:
- Extracting backbone features from different frame subsets
- Merging features from multiple runs with token averaging
- Running prediction head with pre-computed features
"""

from typing import List, Tuple
import torch
from addict import Dict


def extract_backbone_features(
    model,
    images: torch.Tensor,
    ref_view_strategy: str = "first",
) -> Tuple[List, int, int]:
    """
    Extract backbone features from images.

    Args:
        model: DepthAnything3 API instance (or its internal model)
        images: Input images tensor [B, S, 3, H, W]
        ref_view_strategy: Strategy for reference view selection

    Returns:
        Tuple of (feats, H, W) where feats is list of (features, camera_token) tuples
    """
    # Get the internal model if using API wrapper
    net = model.model if hasattr(model, 'model') else model

    # Determine optimal autocast dtype
    device = images.device
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            feats, aux_feats, H, W = net.forward_backbone_only(
                images,
                extrinsics=None,
                intrinsics=None,
                ref_view_strategy=ref_view_strategy,
            )

    return feats, H, W


def run_head_with_features(
    model,
    feats: List,
    H: int,
    W: int,
    process_camera: bool = True,
    process_sky: bool = True,
) -> Dict:
    """
    Run prediction head with pre-computed features.

    Args:
        model: DepthAnything3 API instance (or its internal model)
        feats: List of (features, camera_token) tuples from backbone
        H: Original image height
        W: Original image width
        process_camera: Whether to process camera estimation
        process_sky: Whether to process sky estimation

    Returns:
        Dictionary containing predictions
    """
    # Get the internal model if using API wrapper
    net = model.model if hasattr(model, 'model') else model

    # Get device from features
    device = feats[0][0].device
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            output = net.forward_head_only(
                feats, H, W,
                process_camera=process_camera,
                process_sky=process_sky,
            )

    return output


def merge_features_sequential(
    feat_even: List[Tuple[torch.Tensor, torch.Tensor]],
    feat_odd: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Merge features from even [0,2,4,6] and odd [0,1,3,5,7] frame runs.

    The merging process:
    1. Average frame 0 tokens (both patch features and camera tokens)
    2. Reorder remaining frames to sequential order [0,1,2,3,4,5,6,7]

    Frame mapping:
    - Frame 0: averaged from both runs
    - Frame 1: from odd run (index 1)
    - Frame 2: from even run (index 1)
    - Frame 3: from odd run (index 2)
    - Frame 4: from even run (index 2)
    - Frame 5: from odd run (index 3)
    - Frame 6: from even run (index 3)
    - Frame 7: from odd run (index 4)

    Args:
        feat_even: Features from [0,2,4,6] frames - list of (features, camera_token) tuples
                   features shape: [B, 4, P, C], camera_token shape: [B, 4, D]
        feat_odd: Features from [0,1,3,5,7] frames - list of (features, camera_token) tuples
                  features shape: [B, 5, P, C], camera_token shape: [B, 5, D]

    Returns:
        Merged features as list of (features, camera_token) tuples
        features shape: [B, 8, P, C], camera_token shape: [B, 8, D]
    """
    merged = []

    for (even_feat, even_cam), (odd_feat, odd_cam) in zip(feat_even, feat_odd):
        # even_feat: [B, 4, P, C] for frames [0, 2, 4, 6]
        # odd_feat: [B, 5, P, C] for frames [0, 1, 3, 5, 7]
        # even_cam: [B, 4, D]
        # odd_cam: [B, 5, D]

        # Average frame 0 features
        frame0_feat = (even_feat[:, 0] + odd_feat[:, 0]) / 2  # [B, P, C]
        frame0_cam = (even_cam[:, 0] + odd_cam[:, 0]) / 2  # [B, D]

        # Interleave remaining frames to get sequential order [0,1,2,3,4,5,6,7]
        # even has frames at positions 2,4,6 (indices 1,2,3)
        # odd has frames at positions 1,3,5,7 (indices 1,2,3,4)
        merged_feat = torch.stack([
            frame0_feat,        # 0: averaged
            odd_feat[:, 1],     # 1: from odd
            even_feat[:, 1],    # 2: from even
            odd_feat[:, 2],     # 3: from odd
            even_feat[:, 2],    # 4: from even
            odd_feat[:, 3],     # 5: from odd
            even_feat[:, 3],    # 6: from even
            odd_feat[:, 4],     # 7: from odd
        ], dim=1)  # [B, 8, P, C]

        merged_cam = torch.stack([
            frame0_cam,         # 0: averaged
            odd_cam[:, 1],      # 1: from odd
            even_cam[:, 1],     # 2: from even
            odd_cam[:, 2],      # 3: from odd
            even_cam[:, 2],     # 4: from even
            odd_cam[:, 3],      # 5: from odd
            even_cam[:, 3],     # 6: from even
            odd_cam[:, 4],      # 7: from odd
        ], dim=1)  # [B, 8, D]

        merged.append((merged_feat, merged_cam))

    return merged


def select_frame_subset(
    images: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    indices: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select a subset of frames from the input tensors.

    Args:
        images: Input images [B, S, 3, H, W] or [S, 3, H, W]
        extrinsics: Camera extrinsics [B, S, 4, 4] or [S, 4, 4]
        intrinsics: Camera intrinsics [B, S, 3, 3] or [S, 3, 3]
        indices: List of frame indices to select

    Returns:
        Tuple of (images, extrinsics, intrinsics) with selected frames
    """
    # Handle both batched and unbatched inputs
    if images.dim() == 4:
        # Unbatched: [S, 3, H, W]
        images_subset = images[indices]
        extrinsics_subset = extrinsics[indices]
        intrinsics_subset = intrinsics[indices]
    else:
        # Batched: [B, S, 3, H, W]
        images_subset = images[:, indices]
        extrinsics_subset = extrinsics[:, indices]
        intrinsics_subset = intrinsics[:, indices]

    return images_subset, extrinsics_subset, intrinsics_subset


class TokenAveragingExperiment:
    """
    Handles the token averaging experiment workflow.

    This class manages:
    1. Running backbone on even frames [0,2,4,6]
    2. Running backbone on odd frames [0,1,3,5,7]
    3. Merging features with frame 0 averaging
    4. Running prediction head on merged features
    """

    def __init__(self, model):
        """
        Initialize the experiment.

        Args:
            model: DepthAnything3 API instance
        """
        self.model = model
        self.net = model.model if hasattr(model, 'model') else model

    def run(
        self,
        images: torch.Tensor,
        ref_view_strategy: str = "first",
    ) -> Dict:
        """
        Run the token averaging experiment.

        Args:
            images: Input images [B, 8, 3, H, W] - must have exactly 8 frames
            ref_view_strategy: Strategy for reference view selection

        Returns:
            Dictionary containing predictions from merged features
        """
        B, S, C, H, W = images.shape
        assert S == 8, f"Expected 8 frames, got {S}"

        # Define frame indices
        even_indices = [0, 2, 4, 6]  # 4 frames
        odd_indices = [0, 1, 3, 5, 7]  # 5 frames

        # Select frame subsets
        images_even = images[:, even_indices]  # [B, 4, 3, H, W]
        images_odd = images[:, odd_indices]    # [B, 5, 3, H, W]

        # Extract features from both subsets
        feats_even, H_out, W_out = extract_backbone_features(
            self.model, images_even, ref_view_strategy
        )
        feats_odd, _, _ = extract_backbone_features(
            self.model, images_odd, ref_view_strategy
        )

        # Merge features with frame 0 averaging
        feats_merged = merge_features_sequential(feats_even, feats_odd)

        # Run prediction head on merged features
        output = run_head_with_features(
            self.model, feats_merged, H_out, W_out,
            process_camera=True, process_sky=True
        )

        return output

    def run_with_scene_data(
        self,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        ref_view_strategy: str = "first",
    ) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        Run the token averaging experiment with full scene data.

        This version also handles extrinsics/intrinsics for evaluation.

        Args:
            images: Input images [B, 8, 3, H, W]
            extrinsics: Camera extrinsics [B, 8, 4, 4]
            intrinsics: Camera intrinsics [B, 8, 3, 3]
            ref_view_strategy: Strategy for reference view selection

        Returns:
            Tuple of (output, merged_extrinsics, merged_intrinsics)
        """
        # Run the main experiment
        output = self.run(images, ref_view_strategy)

        # For evaluation, we need to provide the GT extrinsics/intrinsics
        # in the same order as the merged features
        # The merged order is [0,1,2,3,4,5,6,7] which is the original order
        # So we can use the original extrinsics/intrinsics directly

        return output, extrinsics, intrinsics
