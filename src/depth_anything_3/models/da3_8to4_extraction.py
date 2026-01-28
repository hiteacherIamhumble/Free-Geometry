"""
Custom DA3 Model with 8->4 Token Extraction.

This module provides a wrapper around the standard DA3 model that:
1. Takes 8 frames as input
2. Processes all 8 frames through the backbone
3. Extracts only tokens for frames [0, 2, 4, 6]
4. Decodes with the 4 selected tokens

This allows benchmarking 8->4 token extraction using the standard evaluator.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from depth_anything_3.api import DepthAnything3


class DA3_8to4_Extraction(DepthAnything3):
    """
    DA3 model with 8->4 token extraction.

    This model processes 8 frames through the backbone but only decodes
    tokens for frames [0, 2, 4, 6], effectively testing whether processing
    more frames in the encoder helps even when decoding fewer frames.
    """

    STUDENT_FRAME_INDICES = [0, 2, 4, 6]

    def __init__(self, model_name: str = "da3-giant", **kwargs):
        """Initialize with 8->4 extraction enabled."""
        super().__init__(model_name=model_name, **kwargs)
        print("[8->4 Extraction] Model initialized with token extraction enabled")
        print(f"[8->4 Extraction] Will extract frames {self.STUDENT_FRAME_INDICES} from 8-frame input")

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with 8->4 token extraction.

        Args:
            x: Input images (B, N, 3, H, W) where N=8
            extrinsics: Camera extrinsics (B, N, 4, 4)
            intrinsics: Camera intrinsics (B, N, 3, 3)
            export_feat_layers: Layer indices to extract features from
            infer_gs: Enable Gaussian Splatting branch
            use_ray_pose: Use ray-based pose estimation
            ref_view_strategy: Reference view strategy

        Returns:
            Dictionary containing predictions for 4 frames
        """
        B, N, C, H, W = x.shape

        # Ensure we have 8 frames
        if N != 8:
            # If not 8 frames, fall back to standard forward
            return self.model(
                x, extrinsics, intrinsics, export_feat_layers,
                infer_gs, use_ray_pose, ref_view_strategy
            )

        # Step 1: Run backbone with all 8 frames
        feats, aux_feats, orig_H, orig_W = self.model.forward_backbone_only(
            x,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            ref_view_strategy=ref_view_strategy,
        )

        # Step 2: Extract tokens for frames [0, 2, 4, 6]
        reduced_feats = []
        for feat_tuple in feats:
            features, cam_tokens = feat_tuple

            # Extract features for selected frames
            # features shape: (B, N, num_patches, feat_dim)
            reduced_features = features[:, self.STUDENT_FRAME_INDICES, :, :]

            # Extract camera tokens for selected frames
            # cam_tokens shape: (B, N, feat_dim) - 3D tensor
            if cam_tokens is not None:
                reduced_cam_tokens = cam_tokens[:, self.STUDENT_FRAME_INDICES, :]
            else:
                reduced_cam_tokens = None

            reduced_feats.append((reduced_features, reduced_cam_tokens))

        # Step 3: Run head with reduced features (4 frames)
        output = self.model.forward_head_only(
            reduced_feats,
            H=orig_H,
            W=orig_W,
            process_camera=True,
            process_sky=True,
        )

        return output


# Register the custom model
from depth_anything_3.registry import MODEL_REGISTRY

# Create a config for the 8->4 extraction model
DA3_8TO4_CONFIG = {
    "type": "DA3_8to4_Extraction",
    "model_name": "da3-giant",
}

# Note: To use this model, you would need to:
# 1. Import this module
# 2. Create an instance: model = DA3_8to4_Extraction.from_pretrained("depth-anything/DA3-GIANT-1.1")
# 3. Or modify the evaluator to use this class instead of DepthAnything3
