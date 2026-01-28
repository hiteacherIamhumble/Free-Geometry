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
CLI runner for frame extraction experiments on multiview datasets.

This script runs 4 experiments comparing different frame selection strategies:
1. exp1_baseline: Standard 8-frame pipeline
2. exp2_4frame: 4-frame [0,2,4,6] direct pass
3. exp3_extract4: 8-frame encode -> extract 4-frame features -> predict
4. exp4_spike_ablation: Ablation study on spike dimensions for pose

Supported datasets: eth3d, scannetpp, 7scenes, hiroom, dtu

Usage:
    # ETH3D (default)
    python -m depth_anything_3.bench.experiments.run_experiment \
        --config src/depth_anything_3/bench/experiments/eth3d_frame_experiment.yaml \
        model.path=depth-anything/DA3-GIANT

    # ScanNet++
    python -m depth_anything_3.bench.experiments.run_experiment \
        --dataset scannetpp \
        model.path=depth-anything/DA3-GIANT

    # 7-Scenes
    python -m depth_anything_3.bench.experiments.run_experiment \
        --dataset 7scenes \
        model.path=depth-anything/DA3-GIANT

    # Run specific experiments
    python -m depth_anything_3.bench.experiments.run_experiment \
        --experiment exp1_baseline exp3_extract4 \
        model.path=/path/to/model

    # Run spike ablation experiment
    python -m depth_anything_3.bench.experiments.run_experiment \
        --experiment exp4_spike_ablation \
        --modes spike_ablation \
        model.path=/path/to/model

    # Evaluation only (skip inference)
    python -m depth_anything_3.bench.experiments.run_experiment \
        --eval_only \
        model.path=depth-anything/DA3-GIANT

    # Run on specific scenes
    python -m depth_anything_3.bench.experiments.run_experiment \
        eval.scenes=[courtyard,relief] \
        inference.debug=true
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from addict import Dict as AdictDict
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.experiments.frame_extraction_experiment import (
    Exp1_8FrameBaseline,
    Exp2_4FrameDirect,
    Exp3_8FrameExtract4,
    Exp4_SpikeDimAblation,
    Exp5_TokenReplacement,
    create_experiment,
)
from depth_anything_3.bench.registries import MV_REGISTRY
from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.cfg import load_config
from depth_anything_3.utils.geometry import as_homogeneous
from depth_anything_3.utils.io.input_processor import InputProcessor


class ExperimentRunner:
    """
    Runner for frame extraction experiments.

    Supports multiple datasets:
    - eth3d: ETH3D multiview dataset
    - scannetpp: ScanNet++ indoor dataset

    Handles:
    - Loading model and dataset
    - Running experiments with proper frame selection
    - Saving results in format compatible with Evaluator
    - Computing metrics
    """

    def __init__(
        self,
        work_dir: str,
        experiment_types: List[str],
        model_path: str,
        dataset_name: str = "eth3d",
        ref_view_strategy: str = "first",
        scenes: Optional[List[str]] = None,
        num_frames: int = 8,
        debug: bool = False,
    ):
        """
        Initialize experiment runner.

        Args:
            work_dir: Output directory for results
            experiment_types: List of experiment types to run
            model_path: Path to model checkpoint or HuggingFace ID
            dataset_name: Name of dataset ("eth3d" or "scannetpp")
            ref_view_strategy: Reference view selection strategy
            scenes: Specific scenes to evaluate (None = all)
            num_frames: Number of input frames (must be 8)
            debug: Enable debug output
        """
        self.work_dir = work_dir
        self.experiment_types = experiment_types
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.ref_view_strategy = ref_view_strategy
        self.scenes = scenes
        self.num_frames = num_frames
        self.debug = debug

        # Initialize dataset
        self.dataset = MV_REGISTRY.get(dataset_name)()

        # Input processor for preprocessing images
        self.input_processor = InputProcessor()

        # Model will be loaded lazily
        self._model = None
        self._experiments = {}

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            print(f"[INFO] Loading model from {self.model_path}")

            # Check if loading from local checkpoint file
            if self.model_path.endswith('.pth') or self.model_path.endswith('.pt'):
                # Load base model first
                base_model_path = "depth-anything/DA3-GIANT"
                print(f"[INFO] Loading base model from {base_model_path}")
                self._model = DepthAnything3.from_pretrained(base_model_path)

                # Load adapted weights
                print(f"[INFO] Loading adapted weights from {self.model_path}")
                state_dict = torch.load(self.model_path, map_location='cpu')
                self._model.model.load_state_dict(state_dict)
            else:
                # Load from HuggingFace
                self._model = DepthAnything3.from_pretrained(self.model_path)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(device)
            self._model.eval()
        return self._model

    def get_experiment(self, exp_type: str):
        """Get or create experiment instance."""
        if exp_type not in self._experiments:
            self._experiments[exp_type] = create_experiment(
                exp_type,
                self.model,
                ref_view_strategy=self.ref_view_strategy,
            )
        return self._experiments[exp_type]

    def _get_scenes(self) -> List[str]:
        """Get list of scenes to process."""
        all_scenes = self.dataset.SCENES
        if self.scenes:
            return [s for s in all_scenes if s in self.scenes]
        return all_scenes

    def _export_dir(self, experiment_name: str, scene: str) -> str:
        """Get export directory for experiment/scene."""
        return os.path.join(self.work_dir, experiment_name, self.dataset_name, scene, "unposed")

    def _sample_frames(self, scene_data: AdictDict, scene: str) -> AdictDict:
        """
        Sample exactly num_frames from scene data.
        Uses deterministic sampling for reproducibility.
        """
        total_frames = len(scene_data.image_files)

        if total_frames <= self.num_frames:
            # If fewer frames than needed, pad or handle specially
            if self.debug:
                print(
                    f"[WARNING] Scene {scene} has only {total_frames} frames, "
                    f"expected {self.num_frames}"
                )
            return scene_data

        # Deterministic sampling: select evenly spaced frames
        np.random.seed(42)
        indices = sorted(
            np.random.choice(total_frames, self.num_frames, replace=False).tolist()
        )

        if self.debug:
            print(f"  [Sampling] {scene}: {total_frames} -> {self.num_frames} frames")
            print(f"             Selected indices: {indices}")

        sampled = AdictDict()
        sampled.image_files = [scene_data.image_files[i] for i in indices]
        sampled.extrinsics = scene_data.extrinsics[indices]
        sampled.intrinsics = scene_data.intrinsics[indices]
        sampled.aux = scene_data.aux

        return sampled

    def _preprocess_images(
        self, scene_data: AdictDict
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Load and preprocess images to tensor.

        Returns:
            Tuple of (images_tensor, extrinsics, intrinsics)
        """
        imgs_tensor, ext_tensor, int_tensor = self.input_processor(
            scene_data.image_files,
            scene_data.extrinsics,
            scene_data.intrinsics,
            process_res=504,
            process_res_method="upper_bound_resize",
            num_workers=4,
            print_progress=False,
        )

        device = self.model._get_model_device()
        imgs_tensor = imgs_tensor.to(device).float()  # [S, 3, H, W]

        # Add batch dimension: [1, S, 3, H, W]
        if imgs_tensor.dim() == 4:
            imgs_tensor = imgs_tensor[None]

        return imgs_tensor, ext_tensor, int_tensor

    def _save_results(
        self,
        export_dir: str,
        output: AdictDict,
        gt_scene_data: AdictDict,
        output_indices: Optional[List[int]] = None,
    ):
        """
        Save results in format compatible with Evaluator.

        Creates:
        - exports/mini_npz/results.npz: Model predictions
        - exports/gt_meta.npz: GT data for evaluation

        Args:
            export_dir: Directory to save results
            output: Model output dictionary
            gt_scene_data: Ground truth scene data
            output_indices: Frame indices for output (None = all frames)
        """
        os.makedirs(export_dir, exist_ok=True)

        # Save predictions
        npz_dir = os.path.join(export_dir, "exports", "mini_npz")
        os.makedirs(npz_dir, exist_ok=True)

        # Convert output tensors to numpy
        save_dict = {}

        if "depth" in output:
            depth = output.depth
            if depth.dim() == 4:  # [B, S, H, W]
                depth = depth.squeeze(0)  # [S, H, W]
            save_dict["depth"] = np.round(depth.cpu().numpy(), 8)

        if "depth_conf" in output:
            conf = output.depth_conf
            if conf.dim() == 4:
                conf = conf.squeeze(0)
            save_dict["conf"] = np.round(conf.cpu().numpy(), 2)

        if "extrinsics" in output:
            ext = output.extrinsics
            if ext.dim() == 4:  # [B, S, 3, 4]
                ext = ext.squeeze(0)  # [S, 3, 4]
            save_dict["extrinsics"] = ext.cpu().numpy()

        if "intrinsics" in output:
            ixt = output.intrinsics
            if ixt.dim() == 4:  # [B, S, 3, 3]
                ixt = ixt.squeeze(0)  # [S, 3, 3]
            save_dict["intrinsics"] = ixt.cpu().numpy()

        np.savez_compressed(os.path.join(npz_dir, "results.npz"), **save_dict)

        # Save GT meta (for evaluation)
        # If output_indices is provided, save only those frames' GT
        if output_indices is not None:
            gt_ext = gt_scene_data.extrinsics[output_indices]
            gt_int = gt_scene_data.intrinsics[output_indices]
            gt_files = [gt_scene_data.image_files[i] for i in output_indices]
        else:
            gt_ext = gt_scene_data.extrinsics
            gt_int = gt_scene_data.intrinsics
            gt_files = gt_scene_data.image_files

        meta_dir = os.path.join(export_dir, "exports")
        np.savez_compressed(
            os.path.join(meta_dir, "gt_meta.npz"),
            extrinsics=gt_ext,
            intrinsics=gt_int,
            image_files=np.array(gt_files, dtype=object),
        )

    def _save_spike_ablation_results(
        self,
        export_dir: str,
        output: AdictDict,
        gt_scene_data: AdictDict,
        output_indices: List[int],
    ):
        """
        Save results for spike ablation experiment (exp4).

        Saves separate files for each ablation condition:
        - results_baseline.npz
        - results_spike_fixed.npz
        - results_nonspike_fixed.npz
        - results_teacher.npz
        - spike_dims.npz (spike dimension metadata)

        Args:
            export_dir: Directory to save results
            output: Model output with ablation conditions and spike_info
            gt_scene_data: Ground truth scene data
            output_indices: Frame indices for output
        """
        os.makedirs(export_dir, exist_ok=True)
        npz_dir = os.path.join(export_dir, "exports", "mini_npz")
        os.makedirs(npz_dir, exist_ok=True)

        # Save each ablation condition
        conditions = ["baseline", "spike_fixed", "nonspike_fixed", "first_half_fixed", "last_half_fixed", "student_copy_first_to_last", "student_copy_last_to_first", "teacher"]
        for condition in conditions:
            if condition not in output:
                continue

            cond_output = output[condition]
            save_dict = {}

            if "depth" in cond_output:
                depth = cond_output.depth
                if depth.dim() == 4:
                    depth = depth.squeeze(0)
                save_dict["depth"] = np.round(depth.cpu().numpy(), 8)

            if "depth_conf" in cond_output:
                conf = cond_output.depth_conf
                if conf.dim() == 4:
                    conf = conf.squeeze(0)
                save_dict["conf"] = np.round(conf.cpu().numpy(), 2)

            if "extrinsics" in cond_output:
                ext = cond_output.extrinsics
                if ext.dim() == 4:
                    ext = ext.squeeze(0)
                save_dict["extrinsics"] = ext.cpu().numpy()

            if "intrinsics" in cond_output:
                ixt = cond_output.intrinsics
                if ixt.dim() == 4:
                    ixt = ixt.squeeze(0)
                save_dict["intrinsics"] = ixt.cpu().numpy()

            np.savez_compressed(
                os.path.join(npz_dir, f"results_{condition}.npz"), **save_dict
            )

        # Save spike dimension metadata
        if "spike_info" in output:
            spike_info = output.spike_info
            np.savez_compressed(
                os.path.join(npz_dir, "spike_dims.npz"),
                spike_dims=spike_info.spike_dims.numpy(),
                non_spike_dims=spike_info.non_spike_dims.numpy(),
                per_dim_mse=spike_info.per_dim_mse.numpy(),
                num_spike_dims=spike_info.num_spike_dims,
                num_non_spike_dims=spike_info.num_non_spike_dims,
                spike_percentile=spike_info.spike_percentile,
            )

        # Save GT meta
        gt_ext = gt_scene_data.extrinsics[output_indices]
        gt_int = gt_scene_data.intrinsics[output_indices]
        gt_files = [gt_scene_data.image_files[i] for i in output_indices]

        meta_dir = os.path.join(export_dir, "exports")
        np.savez_compressed(
            os.path.join(meta_dir, "gt_meta.npz"),
            extrinsics=gt_ext,
            intrinsics=gt_int,
            image_files=np.array(gt_files, dtype=object),
        )

    def _save_token_replacement_results(
        self,
        export_dir: str,
        output: AdictDict,
        gt_scene_data: AdictDict,
        output_indices: List[int],
    ):
        """
        Save results for token replacement experiment (exp5).

        Saves separate files for each condition:
        - results_baseline.npz
        - results_teacher.npz
        - results_replace_high_Xpct.npz
        - results_replace_low_Xpct.npz
        - mse_info.npz (MSE statistics)
        """
        os.makedirs(export_dir, exist_ok=True)
        npz_dir = os.path.join(export_dir, "exports", "mini_npz")
        os.makedirs(npz_dir, exist_ok=True)

        # Get all conditions from output
        conditions = [k for k in output.keys() if k not in ["mse_info"]]

        for condition in conditions:
            if condition not in output:
                continue

            cond_output = output[condition]
            save_dict = {}

            if "depth" in cond_output:
                depth = cond_output.depth
                if depth.dim() == 4:
                    depth = depth.squeeze(0)
                save_dict["depth"] = np.round(depth.cpu().numpy(), 8)

            if "depth_conf" in cond_output:
                conf = cond_output.depth_conf
                if conf.dim() == 4:
                    conf = conf.squeeze(0)
                save_dict["conf"] = np.round(conf.cpu().numpy(), 2)

            if "extrinsics" in cond_output:
                ext = cond_output.extrinsics
                if ext.dim() == 4:
                    ext = ext.squeeze(0)
                save_dict["extrinsics"] = ext.cpu().numpy()

            if "intrinsics" in cond_output:
                ixt = cond_output.intrinsics
                if ixt.dim() == 4:
                    ixt = ixt.squeeze(0)
                save_dict["intrinsics"] = ixt.cpu().numpy()

            np.savez_compressed(
                os.path.join(npz_dir, f"results_{condition}.npz"), **save_dict
            )

        # Save MSE info
        if "mse_info" in output:
            mse_info = output.mse_info
            np.savez_compressed(
                os.path.join(npz_dir, "mse_info.npz"),
                mean=mse_info.mean,
                std=mse_info.std,
                min=mse_info.min,
                max=mse_info.max,
                median=mse_info.median,
            )

        # Save GT meta
        gt_ext = gt_scene_data.extrinsics[output_indices]
        gt_int = gt_scene_data.intrinsics[output_indices]
        gt_files = [gt_scene_data.image_files[i] for i in output_indices]

        meta_dir = os.path.join(export_dir, "exports")
        np.savez_compressed(
            os.path.join(meta_dir, "gt_meta.npz"),
            extrinsics=gt_ext,
            intrinsics=gt_int,
            image_files=np.array(gt_files, dtype=object),
        )

    def run_inference(self):
        """Run inference for all experiments and scenes."""
        scenes = self._get_scenes()

        for exp_type in self.experiment_types:
            experiment = self.get_experiment(exp_type)
            print(f"\n{'='*60}")
            print(f"Running experiment: {experiment.experiment_name}")
            print(f"{'='*60}")

            # Check if this is the spike ablation experiment
            is_spike_ablation = isinstance(experiment, Exp4_SpikeDimAblation)
            is_token_replacement = isinstance(experiment, Exp5_TokenReplacement)

            for scene in tqdm(scenes, desc=f"{exp_type}"):
                # Get and sample scene data
                scene_data = self.dataset.get_data(scene)
                sampled_data = self._sample_frames(scene_data, scene)

                # Preprocess images
                images, ext_tensor, int_tensor = self._preprocess_images(sampled_data)

                # Run experiment
                raw_output = experiment.run_inference(images)

                # Determine which GT frames to save
                output_indices = experiment.get_output_frames()

                # Save results
                export_dir = self._export_dir(experiment.experiment_name, scene)

                if is_spike_ablation:
                    # Special handling for spike ablation: save multiple files
                    self._save_spike_ablation_results(
                        export_dir, raw_output, sampled_data, output_indices
                    )
                    if self.debug:
                        spike_info = raw_output.spike_info
                        print(
                            f"  {scene}: {spike_info.num_spike_dims} spike dims "
                            f"(top {spike_info.spike_percentile}%)"
                        )
                elif is_token_replacement:
                    # Special handling for token replacement: save multiple files
                    self._save_token_replacement_results(
                        export_dir, raw_output, sampled_data, output_indices
                    )
                    if self.debug:
                        mse_info = raw_output.mse_info
                        print(
                            f"  {scene}: MSE mean={mse_info.mean:.4f}, "
                            f"std={mse_info.std:.4f}"
                        )
                else:
                    # Standard single-file save
                    self._save_results(export_dir, raw_output, sampled_data, output_indices)
                    if self.debug:
                        depth_shape = raw_output.depth.shape if "depth" in raw_output else "N/A"
                        ext_shape = (
                            raw_output.extrinsics.shape
                            if "extrinsics" in raw_output
                            else "N/A"
                        )
                        print(f"  {scene}: depth shape {depth_shape}, ext shape {ext_shape}")

    def eval_pose(self) -> Dict[str, Dict]:
        """Evaluate pose metrics for all experiments."""
        results = {}
        scenes = self._get_scenes()

        for exp_type in self.experiment_types:
            experiment = self.get_experiment(exp_type)
            exp_name = experiment.experiment_name

            print(f"\n[POSE] Evaluating {exp_name}")
            exp_results = AdictDict()

            for scene in scenes:
                export_dir = self._export_dir(exp_name, scene)
                result_path = os.path.join(
                    export_dir, "exports", "mini_npz", "results.npz"
                )
                meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")

                if not os.path.exists(result_path):
                    print(f"  [SKIP] {scene}: results not found")
                    continue

                try:
                    # Load predictions
                    pred = np.load(result_path)
                    gt_meta = np.load(meta_path)

                    # Compute pose metrics
                    metrics = compute_pose(
                        torch.from_numpy(as_homogeneous(pred["extrinsics"])),
                        torch.from_numpy(as_homogeneous(gt_meta["extrinsics"])),
                    )

                    exp_results[scene] = {k: float(v) for k, v in metrics.items()}
                    if self.debug:
                        print(f"  {scene}: AUC@30={metrics.auc30:.4f}")

                except Exception as e:
                    print(f"  [ERROR] {scene}: {e}")
                    continue

            # Compute mean
            if exp_results:
                keys = list(list(exp_results.values())[0].keys())
                exp_results["mean"] = {
                    k: float(np.mean([r[k] for r in exp_results.values() if k in r]))
                    for k in keys
                }

            results[f"{self.dataset_name}_{exp_name}_pose"] = exp_results

            # Save metrics
            metrics_dir = os.path.join(self.work_dir, exp_name, "metric_results")
            os.makedirs(metrics_dir, exist_ok=True)
            with open(os.path.join(metrics_dir, f"{self.dataset_name}_pose.json"), "w") as f:
                json.dump(dict(exp_results), f, indent=2)

        return results

    def eval_recon_unposed(self) -> Dict[str, Dict]:
        """
        Evaluate 3D reconstruction metrics (recon_unposed) for all experiments.

        This performs:
        1. TSDF fusion of predicted depths using predicted poses
        2. Evaluation against GT mesh (Accuracy, Completeness, F-score)
        """
        results = {}
        scenes = self._get_scenes()

        for exp_type in self.experiment_types:
            experiment = self.get_experiment(exp_type)
            exp_name = experiment.experiment_name

            print(f"\n[RECON_UNPOSED] Evaluating {exp_name}")
            exp_results = AdictDict()

            for scene in tqdm(scenes, desc=f"{exp_name} fusion"):
                export_dir = self._export_dir(exp_name, scene)
                result_path = os.path.join(
                    export_dir, "exports", "mini_npz", "results.npz"
                )
                fuse_path = os.path.join(export_dir, "exports", "fuse", "pcd.ply")

                if not os.path.exists(result_path):
                    print(f"  [SKIP] {scene}: results not found")
                    continue

                try:
                    # Step 1: TSDF fusion
                    self.dataset.fuse3d(scene, result_path, fuse_path, mode="recon_unposed")

                    # Step 2: Evaluate against GT mesh
                    metrics = self.dataset.eval3d(scene, fuse_path)

                    # Add overall metric (mean of acc and comp)
                    scene_results = {k: float(v) for k, v in metrics.items()}
                    scene_results['overall'] = (scene_results['acc'] + scene_results['comp']) / 2.0
                    exp_results[scene] = scene_results
                    print(f"  {scene}: Acc={metrics['acc']:.4f}, Comp={metrics['comp']:.4f}, Overall={scene_results['overall']:.4f}, F={metrics['fscore']:.4f}")

                except Exception as e:
                    print(f"  [ERROR] {scene}: {e}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                    continue

            # Compute mean
            if exp_results:
                keys = list(list(exp_results.values())[0].keys())
                exp_results["mean"] = {
                    k: float(np.mean([r[k] for r in exp_results.values() if k in r]))
                    for k in keys
                }

            results[f"{self.dataset_name}_{exp_name}_recon_unposed"] = exp_results

            # Save metrics
            metrics_dir = os.path.join(self.work_dir, exp_name, "metric_results")
            os.makedirs(metrics_dir, exist_ok=True)
            with open(os.path.join(metrics_dir, f"{self.dataset_name}_recon_unposed.json"), "w") as f:
                json.dump(dict(exp_results), f, indent=2)

        return results

    def eval_spike_ablation(self) -> Dict[str, Dict]:
        """
        Evaluate spike dimension ablation experiment (exp4).

        Computes pose metrics for each ablation condition and calculates
        the "gap closed" metric to determine spike dimension importance.

        Returns:
            Dict with metrics for each scene and condition, plus gap analysis
        """
        results = {}
        scenes = self._get_scenes()

        # Only process exp4_spike_ablation
        exp_type = "exp4_spike_ablation"
        if exp_type not in self.experiment_types:
            print("[SPIKE_ABLATION] exp4_spike_ablation not in experiment types, skipping")
            return results

        experiment = self.get_experiment(exp_type)
        exp_name = experiment.experiment_name
        conditions = ["baseline", "spike_fixed", "nonspike_fixed", "first_half_fixed", "last_half_fixed", "student_copy_first_to_last", "student_copy_last_to_first", "teacher"]

        print(f"\n[SPIKE_ABLATION] Evaluating {exp_name}")
        all_scene_results = AdictDict()

        for scene in scenes:
            export_dir = self._export_dir(exp_name, scene)
            meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")
            spike_dims_path = os.path.join(export_dir, "exports", "mini_npz", "spike_dims.npz")

            # Check if files exist
            if not os.path.exists(meta_path):
                print(f"  [SKIP] {scene}: GT meta not found")
                continue

            try:
                gt_meta = np.load(meta_path)
                gt_ext = torch.from_numpy(as_homogeneous(gt_meta["extrinsics"]))

                # Load spike dim info if available
                spike_info = {}
                if os.path.exists(spike_dims_path):
                    spike_data = np.load(spike_dims_path)
                    spike_info = {
                        "num_spike_dims": int(spike_data["num_spike_dims"]),
                        "spike_dims": spike_data["spike_dims"].tolist(),
                    }

                scene_results = AdictDict()
                scene_results.spike_info = spike_info

                # Evaluate each condition
                for condition in conditions:
                    result_path = os.path.join(
                        export_dir, "exports", "mini_npz", f"results_{condition}.npz"
                    )

                    if not os.path.exists(result_path):
                        print(f"  [SKIP] {scene}/{condition}: results not found")
                        continue

                    pred = np.load(result_path)
                    pred_ext = torch.from_numpy(as_homogeneous(pred["extrinsics"]))

                    # Compute pose metrics
                    metrics = compute_pose(pred_ext, gt_ext)
                    scene_results[condition] = {
                        k: float(v) for k, v in metrics.items()
                    }

                # Compute gap closed metric
                if all(c in scene_results for c in ["baseline", "spike_fixed", "teacher"]):
                    baseline_err = scene_results.baseline.get("auc30", 0)
                    spike_fixed_err = scene_results.spike_fixed.get("auc30", 0)
                    teacher_err = scene_results.teacher.get("auc30", 0)

                    # Gap closed = (baseline - spike_fixed) / (baseline - teacher)
                    # Higher AUC is better, so improvement is spike_fixed - baseline
                    if teacher_err > baseline_err:  # Teacher is better
                        gap = teacher_err - baseline_err
                        improvement = spike_fixed_err - baseline_err
                        gap_closed = improvement / gap if gap > 1e-8 else 0.0
                    else:
                        gap_closed = 0.0

                    scene_results.gap_closed = float(gap_closed)

                    # Also compute for nonspike (control)
                    if "nonspike_fixed" in scene_results:
                        nonspike_err = scene_results.nonspike_fixed.get("auc30", 0)
                        improvement_nonspike = nonspike_err - baseline_err
                        gap_closed_nonspike = improvement_nonspike / gap if gap > 1e-8 else 0.0
                        scene_results.gap_closed_nonspike = float(gap_closed_nonspike)

                all_scene_results[scene] = scene_results

                if self.debug:
                    print(
                        f"  {scene}: baseline={baseline_err:.4f}, "
                        f"spike_fixed={spike_fixed_err:.4f}, "
                        f"teacher={teacher_err:.4f}, "
                        f"gap_closed={gap_closed*100:.1f}%"
                    )

            except Exception as e:
                print(f"  [ERROR] {scene}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                continue

        # Compute aggregated statistics
        if all_scene_results:
            # Get only scene results (exclude aggregate keys)
            scene_results_only = {
                k: v for k, v in all_scene_results.items()
                if isinstance(v, (dict, AdictDict)) and not k.startswith("mean_")
            }

            # Mean per condition
            for condition in conditions:
                cond_results = [
                    r[condition] for r in scene_results_only.values()
                    if condition in r and isinstance(r[condition], dict)
                ]
                if cond_results:
                    keys = list(cond_results[0].keys())
                    all_scene_results[f"mean_{condition}"] = {
                        k: float(np.mean([r[k] for r in cond_results if k in r]))
                        for k in keys
                    }

            # Mean gap closed
            gap_values = [
                r.gap_closed for r in scene_results_only.values()
                if "gap_closed" in r
            ]
            if gap_values:
                all_scene_results["mean_gap_closed"] = float(np.mean(gap_values))
                all_scene_results["std_gap_closed"] = float(np.std(gap_values))

            gap_nonspike_values = [
                r.gap_closed_nonspike for r in scene_results_only.values()
                if "gap_closed_nonspike" in r
            ]
            if gap_nonspike_values:
                all_scene_results["mean_gap_closed_nonspike"] = float(np.mean(gap_nonspike_values))

            # Also compute gap closed from mean AUC values (more interpretable)
            # Compute for multiple AUC thresholds
            auc_metrics = ["auc30", "auc15", "auc05", "auc03"]
            ablation_prefixes = ["spike", "nonspike", "first_half", "last_half", "copy_first_to_last", "copy_last_to_first"]
            condition_map = {
                "spike": "spike_fixed",
                "nonspike": "nonspike_fixed",
                "first_half": "first_half_fixed",
                "last_half": "last_half_fixed",
                "copy_first_to_last": "student_copy_first_to_last",
                "copy_last_to_first": "student_copy_last_to_first",
            }

            if all(f"mean_{c}" in all_scene_results for c in ["baseline", "teacher"]):
                for auc_metric in auc_metrics:
                    mean_baseline = all_scene_results["mean_baseline"].get(auc_metric, 0)
                    mean_teacher = all_scene_results["mean_teacher"].get(auc_metric, 0)
                    gap = mean_teacher - mean_baseline

                    for prefix in ablation_prefixes:
                        condition = condition_map[prefix]
                        mean_key = f"mean_{condition}"
                        if mean_key in all_scene_results and gap > 1e-8:
                            mean_fixed = all_scene_results[mean_key].get(auc_metric, 0)
                            all_scene_results[f"gap_closed_{prefix}_{auc_metric}"] = float(
                                (mean_fixed - mean_baseline) / gap
                            )

        results[f"eth3d_{exp_name}"] = all_scene_results

        # Save metrics
        metrics_dir = os.path.join(self.work_dir, exp_name, "metric_results")
        os.makedirs(metrics_dir, exist_ok=True)
        with open(os.path.join(metrics_dir, "eth3d_spike_ablation.json"), "w") as f:
            # Convert AdictDict to regular dict for JSON serialization
            json_results = {}
            for k, v in all_scene_results.items():
                if isinstance(v, AdictDict):
                    json_results[k] = dict(v)
                else:
                    json_results[k] = v
            json.dump(json_results, f, indent=2)

        # Print summary
        self._print_spike_ablation_summary(all_scene_results)

        return results

    def _print_spike_ablation_summary(self, results: AdictDict):
        """Print summary of spike ablation results."""
        print("\n" + "=" * 85)
        print("SPIKE DIMENSION ABLATION RESULTS")
        print("=" * 85)

        # Print per-condition AUC metrics
        print(f"\n{'Condition':<30} {'AUC@30':>10} {'AUC@15':>10} {'AUC@5':>10} {'AUC@3':>10}")
        print("-" * 70)

        conditions = ["baseline", "spike_fixed", "nonspike_fixed", "first_half_fixed", "last_half_fixed", "student_copy_first_to_last", "student_copy_last_to_first", "teacher"]
        for condition in conditions:
            mean_key = f"mean_{condition}"
            if mean_key in results:
                m = results[mean_key]
                print(
                    f"{condition:<30} {m.get('auc30', 0):>10.4f} "
                    f"{m.get('auc15', 0):>10.4f} {m.get('auc05', 0):>10.4f} "
                    f"{m.get('auc03', 0):>10.4f}"
                )

        # Print gap analysis for all metrics
        print("\n" + "-" * 85)
        print("GAP ANALYSIS (% of baseline-to-teacher gap closed)")
        print("-" * 85)

        print(f"\n{'Ablation':<35} {'AUC@30':>10} {'AUC@15':>10} {'AUC@5':>10} {'AUC@3':>10}")
        print("-" * 75)

        # All ablation conditions
        ablation_rows = [
            ("Spike dims fixed (3%)", "spike"),
            ("Non-spike dims fixed (97%)", "nonspike"),
            ("First half [0:768] w/ teacher", "first_half"),
            ("Last half [768:1536] w/ teacher", "last_half"),
            ("Student: copy 1st->2nd half", "copy_first_to_last"),
            ("Student: copy 2nd->1st half", "copy_last_to_first"),
        ]

        for label, prefix in ablation_rows:
            gaps = []
            for metric in ["auc30", "auc15", "auc05", "auc03"]:
                key = f"gap_closed_{prefix}_{metric}"
                val = results.get(key, 0) * 100
                gaps.append(f"{val:>10.1f}%")
            print(f"{label:<35} " + " ".join(gaps))

        # Interpretation
        print("\n" + "-" * 60)
        print("INTERPRETATION")
        print("-" * 60)

        gap_closed = results.get("gap_closed_spike_auc30", results.get("mean_gap_closed", 0))
        if gap_closed is not None:
            if gap_closed > 0.5:
                print("Result: Spike dims ARE KEY for pose estimation")
                print("Action: Focus distillation loss on these dimensions")
            elif gap_closed < 0.2:
                print("Result: Spike dims are NOT KEY for pose estimation")
                print("Action: Decoder ignores them; try other alignment strategies")
            else:
                print("Result: Spike dims are PARTIALLY IMPORTANT")
                print("Action: Include but don't over-weight in loss")

        print("=" * 80)

    def eval_token_replacement(self) -> Dict[str, Dict]:
        """
        Evaluate token replacement experiment (exp5).

        Computes pose metrics for each replacement condition.
        """
        results = {}
        scenes = self._get_scenes()

        exp_type = "exp5_token_replacement"
        if exp_type not in self.experiment_types:
            print("[TOKEN_REPLACEMENT] exp5_token_replacement not in experiment types")
            return results

        experiment = self.get_experiment(exp_type)
        exp_name = experiment.experiment_name

        # All conditions to evaluate
        conditions = ["baseline", "teacher"]
        for pct in [10, 20, 30, 50]:
            conditions.append(f"replace_high_{pct}pct")
            conditions.append(f"replace_low_{100-pct}pct")

        print(f"\n[TOKEN_REPLACEMENT] Evaluating {exp_name}")
        all_scene_results = AdictDict()

        for scene in scenes:
            export_dir = self._export_dir(exp_name, scene)
            meta_path = os.path.join(export_dir, "exports", "gt_meta.npz")

            if not os.path.exists(meta_path):
                print(f"  [SKIP] {scene}: GT meta not found")
                continue

            try:
                gt_meta = np.load(meta_path)
                gt_ext = torch.from_numpy(as_homogeneous(gt_meta["extrinsics"]))

                scene_results = AdictDict()

                for condition in conditions:
                    result_path = os.path.join(
                        export_dir, "exports", "mini_npz", f"results_{condition}.npz"
                    )

                    if not os.path.exists(result_path):
                        continue

                    pred = np.load(result_path)
                    pred_ext = torch.from_numpy(as_homogeneous(pred["extrinsics"]))

                    metrics = compute_pose(pred_ext, gt_ext)
                    scene_results[condition] = {k: float(v) for k, v in metrics.items()}

                all_scene_results[scene] = scene_results

            except Exception as e:
                print(f"  [ERROR] {scene}: {e}")
                continue

        # Compute mean per condition
        if all_scene_results:
            scene_results_only = {
                k: v for k, v in all_scene_results.items()
                if isinstance(v, (dict, AdictDict)) and not k.startswith("mean_")
            }

            for condition in conditions:
                cond_results = [
                    r[condition] for r in scene_results_only.values()
                    if condition in r
                ]
                if cond_results:
                    keys = list(cond_results[0].keys())
                    all_scene_results[f"mean_{condition}"] = {
                        k: float(np.mean([r[k] for r in cond_results]))
                        for k in keys
                    }

        results[f"{self.dataset_name}_{exp_name}"] = all_scene_results

        # Save metrics
        metrics_dir = os.path.join(self.work_dir, exp_name, "metric_results")
        os.makedirs(metrics_dir, exist_ok=True)
        with open(os.path.join(metrics_dir, f"{self.dataset_name}_token_replacement.json"), "w") as f:
            json_results = {}
            for k, v in all_scene_results.items():
                if isinstance(v, AdictDict):
                    json_results[k] = dict(v)
                else:
                    json_results[k] = v
            json.dump(json_results, f, indent=2)

        # Print summary
        self._print_token_replacement_summary(all_scene_results)

        return results

    def _print_token_replacement_summary(self, results: AdictDict):
        """Print summary of token replacement results."""
        print("\n" + "=" * 85)
        print("TOKEN REPLACEMENT EXPERIMENT RESULTS")
        print("=" * 85)

        # Print per-condition AUC metrics
        print(f"\n{'Condition':<25} {'AUC@30':>10} {'AUC@15':>10} {'AUC@5':>10}")
        print("-" * 55)

        conditions = ["baseline", "teacher"]
        for pct in [10, 20, 30, 50]:
            conditions.append(f"replace_high_{pct}pct")
            conditions.append(f"replace_low_{100-pct}pct")

        for condition in conditions:
            mean_key = f"mean_{condition}"
            if mean_key in results:
                m = results[mean_key]
                print(
                    f"{condition:<25} {m.get('auc30', 0):>10.4f} "
                    f"{m.get('auc15', 0):>10.4f} {m.get('auc05', 0):>10.4f}"
                )

        # Gap analysis
        print("\n" + "-" * 85)
        print("GAP ANALYSIS (% of baseline-to-teacher gap closed)")
        print("-" * 85)

        if "mean_baseline" in results and "mean_teacher" in results:
            baseline_auc = results["mean_baseline"].get("auc30", 0)
            teacher_auc = results["mean_teacher"].get("auc30", 0)
            gap = teacher_auc - baseline_auc

            print(f"\nBaseline AUC@30: {baseline_auc:.4f}")
            print(f"Teacher AUC@30:  {teacher_auc:.4f}")
            print(f"Gap: {gap:.4f}")

            if abs(gap) > 1e-6:
                print(f"\n{'Condition':<25} {'Gap Closed':>12}")
                print("-" * 40)

                for pct in [10, 20, 30, 50]:
                    for cond in [f"replace_high_{pct}pct", f"replace_low_{100-pct}pct"]:
                        mean_key = f"mean_{cond}"
                        if mean_key in results:
                            cond_auc = results[mean_key].get("auc30", 0)
                            gap_closed = (cond_auc - baseline_auc) / gap * 100
                            print(f"{cond:<25} {gap_closed:>11.1f}%")

        # Interpretation
        print("\n" + "-" * 60)
        print("INTERPRETATION")
        print("-" * 60)
        print("- If replacing HIGH MSE tokens improves: outliers hurt performance")
        print("- If replacing LOW MSE tokens improves: high-MSE tokens are important")
        print("=" * 85)

    def print_comparison(self, pose_metrics: Dict[str, Dict], recon_metrics: Dict[str, Dict] = None, spike_ablation_metrics: Dict[str, Dict] = None):
        """Print comparison table of all experiments."""
        print("\n" + "=" * 100)
        print("EXPERIMENT COMPARISON - POSE METRICS")
        print("=" * 100)

        # Header for pose
        print(f"{'Experiment':<35} {'AUC@30':>10} {'AUC@15':>10} {'AUC@5':>10}")
        print("-" * 65)

        for exp_name, exp_metrics in pose_metrics.items():
            if "mean" in exp_metrics:
                mean = exp_metrics["mean"]
                print(
                    f"{exp_name:<35} {mean.get('auc30', 0):>10.4f} "
                    f"{mean.get('auc15', 0):>10.4f} {mean.get('auc05', 0):>10.4f}"
                )

        # Reconstruction metrics
        if recon_metrics:
            print("\n" + "=" * 100)
            print("EXPERIMENT COMPARISON - RECONSTRUCTION METRICS (recon_unposed)")
            print("=" * 100)

            # Header for recon
            print(f"{'Experiment':<35} {'Acc':>10} {'Comp':>10} {'Overall':>10} {'F-score':>10}")
            print("-" * 75)

            for exp_name, exp_metrics in recon_metrics.items():
                if "mean" in exp_metrics:
                    mean = exp_metrics["mean"]
                    print(
                        f"{exp_name:<35} {mean.get('acc', 0):>10.4f} "
                        f"{mean.get('comp', 0):>10.4f} {mean.get('overall', 0):>10.4f} "
                        f"{mean.get('fscore', 0):>10.4f}"
                    )

        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Run frame extraction experiments on multiview datasets"
    )
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument(
        "--dataset",
        type=str,
        default="eth3d",
        choices=["eth3d", "scannetpp", "7scenes", "hiroom", "dtu"],
        help="Dataset to use for experiments",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        nargs="+",
        default=["exp1_baseline", "exp2_4frame", "exp3_extract4"],
        help="Experiments to run",
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="Only run evaluation (skip inference)"
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["pose", "recon_unposed"],
        help="Evaluation modes: pose, recon_unposed, spike_ablation (default: pose, recon_unposed)",
    )
    args, remaining = parser.parse_known_args()

    # Load config
    config_path = args.config or os.path.join(
        os.path.dirname(__file__), "eth3d_frame_experiment.yaml"
    )
    config = load_config(config_path, argv=remaining)

    # Parse scenes from config
    scenes = config.eval.scenes
    if scenes is not None:
        scenes = list(scenes)

    # Parse modes: CLI args take precedence, then config, then defaults
    # Check if user explicitly provided --modes by seeing if it differs from default
    cli_modes_provided = args.modes != ["pose", "recon_unposed"]
    if cli_modes_provided:
        modes = args.modes
    elif hasattr(config.eval, "modes") and config.eval.modes:
        modes = list(config.eval.modes)
    else:
        modes = args.modes  # Use CLI default

    runner = ExperimentRunner(
        work_dir=config.workspace.work_dir,
        experiment_types=args.experiment,
        model_path=config.model.path,
        dataset_name=args.dataset,
        ref_view_strategy=config.eval.ref_view_strategy,
        scenes=scenes,
        num_frames=config.eval.num_frames,
        debug=config.inference.debug,
    )

    if not args.eval_only:
        runner.run_inference()

    # Run evaluations based on modes
    pose_metrics = None
    recon_metrics = None
    spike_ablation_metrics = None
    token_replacement_metrics = None

    if "pose" in modes:
        pose_metrics = runner.eval_pose()

    if "recon_unposed" in modes:
        recon_metrics = runner.eval_recon_unposed()

    if "spike_ablation" in modes:
        spike_ablation_metrics = runner.eval_spike_ablation()

    if "token_replacement" in modes:
        token_replacement_metrics = runner.eval_token_replacement()

    # Print comparison
    if pose_metrics:
        runner.print_comparison(pose_metrics, recon_metrics, spike_ablation_metrics)


if __name__ == "__main__":
    main()
