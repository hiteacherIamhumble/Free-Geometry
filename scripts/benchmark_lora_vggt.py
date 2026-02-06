#!/usr/bin/env python3
"""
Benchmark script for LoRA-finetuned VGGT model.

Usage:
    python scripts/benchmark_lora_vggt.py \
        --lora_path checkpoints/vggt_distill/epoch_1_lora.pt \
        --datasets scannetpp \
        --modes pose recon_unposed \
        --max_frames 8
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Add vggt submodule to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'vggt'))

from vggt.vggt.bench.evaluator import VGGTEvaluator
from vggt.vggt.distillation.models import VGGTStudentModel
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class LoRAVGGT:
    """
    Wrapper to make VGGTStudentModel compatible with the VGGTEvaluator API.

    The Evaluator expects an object with an `inference` method that takes
    image files and returns predictions.
    """

    def __init__(
        self,
        base_model: str = "facebook/vggt-1b",
        lora_path: str = None,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_layers_start: int = 12,
        device: str = "cuda",
        image_size: int = 504,
    ):
        print(f"Loading base model: {base_model}")
        print(f"Loading LoRA weights: {lora_path}")

        self.image_size = image_size
        self.device = device

        # Create student model with LoRA
        lora_layers = list(range(lora_layers_start, 24))
        self.student = VGGTStudentModel(
            model_name=base_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_layers=lora_layers,
        )

        # Load LoRA weights
        if lora_path and os.path.exists(lora_path):
            self.student.load_lora_weights(lora_path)
        elif lora_path:
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

        self.student.eval()

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.student = self.student.to(device)
        return self

    def _load_images(self, image_files):
        """Load and preprocess images using DA3-style preprocessing.

        - Resize longest side to image_size (default 504)
        - Preserve aspect ratio
        - Round dimensions to nearest multiple of PATCH_SIZE (14)
        """
        PATCH_SIZE = 14
        images = []

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]

            # Resize longest side to image_size (DA3 upper_bound_resize)
            scale = self.image_size / max(h, w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))

            # Round to nearest multiple of PATCH_SIZE
            new_h = max(PATCH_SIZE, round(new_h / PATCH_SIZE) * PATCH_SIZE)
            new_w = max(PATCH_SIZE, round(new_w / PATCH_SIZE) * PATCH_SIZE)

            # Use appropriate interpolation
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)

            img = img.astype(np.float32) / 255.0
            images.append(img)

        # All images in a scene have the same original size, so stack directly
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        return images.unsqueeze(0)  # Add batch dim: [1, S, 3, H, W]

    def inference(
        self,
        image_files,
        extrinsics=None,
        intrinsics=None,
        **kwargs,
    ):
        """
        Run inference using the LoRA-adapted VGGT model.

        Args:
            image_files: List of image file paths
            extrinsics: Optional GT extrinsics (for posed mode)
            intrinsics: Optional GT intrinsics (for posed mode)

        Returns:
            Dict with predictions (pose_enc, depth, world_points, etc.)
        """
        # Load and preprocess images
        images = self._load_images(image_files).to(self.device)

        # Access the underlying VGGT model through PEFT wrapper
        if hasattr(self.student.vggt, 'base_model'):
            vggt_model = self.student.vggt.base_model.model
        else:
            vggt_model = self.student.vggt

        # Run VGGT forward pass
        with torch.no_grad():
            predictions = vggt_model(images)

        # Convert pose_enc to extrinsics for evaluation
        if "pose_enc" in predictions:
            pose_enc = predictions["pose_enc"]
            if pose_enc.dim() == 2:
                pose_enc = pose_enc.unsqueeze(0)

            # Get actual image dimensions
            _, _, _, H, W = images.shape
            ext, intr = pose_encoding_to_extri_intri(
                pose_enc,
                image_size_hw=(H, W),  # Use actual processed size
                pose_encoding_type="absT_quaR_FoV",
            )
            predictions["extrinsics"] = ext.squeeze(0)  # [S, 3, 4]
            predictions["intrinsics"] = intr.squeeze(0)  # [S, 3, 3]

        return predictions


class BaseVGGT:
    """
    Wrapper for base VGGT model (without LoRA) for comparison.
    """

    def __init__(
        self,
        model_name: str = "facebook/vggt-1b",
        device: str = "cuda",
        image_size: int = 504,
    ):
        print(f"Loading base VGGT model: {model_name}")

        self.image_size = image_size
        self.device = device

        # Load base VGGT model
        self.vggt = VGGT.from_pretrained(model_name)
        self.vggt.eval()

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.vggt = self.vggt.to(device)
        return self

    def _load_images(self, image_files):
        """Load and preprocess images using DA3-style preprocessing.

        - Resize longest side to image_size (default 504)
        - Preserve aspect ratio
        - Round dimensions to nearest multiple of PATCH_SIZE (14)
        """
        PATCH_SIZE = 14
        images = []

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]

            # Resize longest side to image_size (DA3 upper_bound_resize)
            scale = self.image_size / max(h, w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))

            # Round to nearest multiple of PATCH_SIZE
            new_h = max(PATCH_SIZE, round(new_h / PATCH_SIZE) * PATCH_SIZE)
            new_w = max(PATCH_SIZE, round(new_w / PATCH_SIZE) * PATCH_SIZE)

            # Use appropriate interpolation
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)

            img = img.astype(np.float32) / 255.0
            images.append(img)

        # All images in a scene have the same original size, so stack directly
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        return images.unsqueeze(0)

    def inference(
        self,
        image_files,
        extrinsics=None,
        intrinsics=None,
        **kwargs,
    ):
        """Run inference using the base VGGT model."""
        images = self._load_images(image_files).to(self.device)

        with torch.no_grad():
            predictions = self.vggt(images)

        # Convert pose_enc to extrinsics
        if "pose_enc" in predictions:
            pose_enc = predictions["pose_enc"]
            if pose_enc.dim() == 2:
                pose_enc = pose_enc.unsqueeze(0)

            # Get actual image dimensions
            _, _, _, H, W = images.shape
            ext, intr = pose_encoding_to_extri_intri(
                pose_enc,
                image_size_hw=(H, W),  # Use actual processed size
                pose_encoding_type="absT_quaR_FoV",
            )
            predictions["extrinsics"] = ext.squeeze(0)
            predictions["intrinsics"] = intr.squeeze(0)

        return predictions


def main():
    parser = argparse.ArgumentParser(description="Benchmark LoRA-finetuned VGGT model")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights (if None, uses base model)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="facebook/vggt-1b",
        help="Base model name",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (must match training)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16.0,
        help="LoRA alpha (must match training)",
    )
    parser.add_argument(
        "--lora_layers_start",
        type=int,
        default=12,
        help="First layer with LoRA (must match training)",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./workspace/vggt_evaluation",
        help="Output directory",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eth3d"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["pose"],
        help="Evaluation modes (pose, recon_unposed, recon_posed)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Max frames per scene",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Specific scenes to evaluate",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=504,
        help="VGGT input image size (longest side, DA3-style)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for frame sampling",
    )
    parser.add_argument(
        "--eval_frames",
        type=int,
        default=None,
        help="If set, use even-indexed frames from max_frames sample",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation (skip inference)",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Only print saved metrics",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--subset_sampling",
        action="store_true",
        help="Use consecutive window sampling from subset (like training)",
    )
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=0.1,
        help="Ratio of frames to include in subset (default: 0.1 = 10%%)",
    )
    args = parser.parse_args()

    # Create evaluator
    evaluator = VGGTEvaluator(
        work_dir=args.work_dir,
        datas=args.datasets,
        modes=args.modes,
        max_frames=args.max_frames,
        scenes=args.scenes,
        image_size=args.image_size,
        debug=args.debug,
        seed=args.seed,
        eval_frames=args.eval_frames,
        subset_sampling=args.subset_sampling,
        subset_ratio=args.subset_ratio,
    )

    if args.print_only:
        evaluator.print_metrics()
        return

    if args.eval_only:
        metrics = evaluator.eval()
        evaluator.print_metrics(metrics)
        return

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.lora_path:
        # Use LoRA-adapted model
        api = LoRAVGGT(
            base_model=args.base_model,
            lora_path=args.lora_path,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_layers_start=args.lora_layers_start,
            image_size=args.image_size,
        ).to(device)
    else:
        # Use base model
        api = BaseVGGT(
            model_name=args.base_model,
            image_size=args.image_size,
        ).to(device)

    # Run inference and evaluation
    evaluator.infer(api)
    metrics = evaluator.eval()
    evaluator.print_metrics(metrics)


if __name__ == "__main__":
    main()
