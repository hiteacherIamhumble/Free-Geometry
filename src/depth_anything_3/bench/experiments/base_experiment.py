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
Base class for frame extraction experiments.

Provides common utilities for running experiments and saving results.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch
from addict import Dict as AdictDict


class BaseFrameExperiment(ABC):
    """
    Abstract base class for frame extraction experiments.

    Subclasses must implement:
        - run_inference(): Run model inference with experiment-specific logic
        - experiment_name: Property returning the experiment identifier
    """

    def __init__(
        self,
        model,
        ref_view_strategy: str = "first",
    ):
        """
        Initialize base experiment.

        Args:
            model: DepthAnything3 API instance
            ref_view_strategy: Reference view selection strategy
        """
        self.model = model
        self.net = model.model if hasattr(model, "model") else model
        self.ref_view_strategy = ref_view_strategy

    @property
    @abstractmethod
    def experiment_name(self) -> str:
        """Return unique experiment identifier."""
        raise NotImplementedError

    @abstractmethod
    def run_inference(
        self,
        images: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> AdictDict:
        """
        Run experiment-specific inference.

        Args:
            images: Preprocessed images tensor [B, S, 3, H, W]
            extrinsics: Optional GT extrinsics [B, S, 4, 4]
            intrinsics: Optional GT intrinsics [B, S, 3, 3]

        Returns:
            Raw model output dictionary
        """
        raise NotImplementedError

    def get_output_frames(self) -> Optional[List[int]]:
        """
        Return the frame indices that this experiment outputs predictions for.

        Returns:
            List of frame indices, or None to indicate all frames.
        """
        return None  # None means all input frames

    def convert_output_to_numpy(self, output: AdictDict) -> Dict[str, np.ndarray]:
        """
        Convert model output tensors to numpy arrays for saving.

        Args:
            output: Model output dictionary with tensors

        Returns:
            Dictionary with numpy arrays
        """
        result = {}

        if "depth" in output:
            # Remove batch dimension if present
            depth = output.depth
            if depth.dim() == 4:  # [B, S, H, W]
                depth = depth.squeeze(0)  # [S, H, W]
            result["depth"] = depth.cpu().numpy()

        if "depth_conf" in output:
            conf = output.depth_conf
            if conf.dim() == 4:
                conf = conf.squeeze(0)
            result["conf"] = conf.cpu().numpy()

        if "extrinsics" in output:
            ext = output.extrinsics
            if ext.dim() == 4:  # [B, S, 3, 4] or [B, S, 4, 4]
                ext = ext.squeeze(0)  # [S, 3, 4] or [S, 4, 4]
            result["extrinsics"] = ext.cpu().numpy()

        if "intrinsics" in output:
            ixt = output.intrinsics
            if ixt.dim() == 4:  # [B, S, 3, 3]
                ixt = ixt.squeeze(0)  # [S, 3, 3]
            result["intrinsics"] = ixt.cpu().numpy()

        return result
