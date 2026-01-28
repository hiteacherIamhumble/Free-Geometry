#!/usr/bin/env python3
"""
Benchmark script for LoRA-finetuned DepthAnything3 model.

Usage:
    python scripts/train_distill.py \
      --data_root ./data \
      --mse_only \
      --epochs 1 \
      --output_dir ./checkpoints/distill_mse &&
    python scripts/benchmark_lora.py \
        --lora_path checkpoints/distill/best_lora.pt \
        --datasets scannetpp \
        --modes pose recon_unposed \
        --max_frames 4
"""

import argparse
import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator
from depth_anything_3.distillation.models import StudentModel


class LoRADepthAnything3:
    """
    Wrapper to make StudentModel compatible with the Evaluator API.

    The Evaluator expects an object with an `inference` method matching
    the DepthAnything3 API signature.
    """

    def __init__(
        self,
        base_model: str = "depth-anything/DA3-GIANT",
        lora_path: str = None,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        device: str = "cuda",
    ):
        print(f"Loading base model: {base_model}")
        print(f"Loading LoRA weights: {lora_path}")

        # Create student model with LoRA
        self.student = StudentModel(
            model_name=base_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        # Load LoRA weights
        if lora_path and os.path.exists(lora_path):
            self.student.load_lora_weights(lora_path)
        else:
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

        self.student.eval()
        self.device = device

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.student = self.student.to(device)
        return self

    def inference(
        self,
        image,
        extrinsics=None,
        intrinsics=None,
        align_to_input_ext_scale=True,
        export_dir=None,
        export_format="mini_npz",
        ref_view_strategy="first",
        **kwargs,
    ):
        """
        Run inference using the LoRA-adapted model.

        This wraps the underlying DA3 API's inference method.
        """
        # Use the underlying DA3 model's inference
        return self.student.da3.inference(
            image=image,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=align_to_input_ext_scale,
            export_dir=export_dir,
            export_format=export_format,
            ref_view_strategy=ref_view_strategy,
            **kwargs,
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark LoRA-finetuned DA3 model")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="checkpoints/distill/best_lora.pt",
        help="Path to LoRA weights",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="depth-anything/DA3-GIANT",
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
        "--work_dir",
        type=str,
        default="./workspace/evaluation_lora",
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
        default=["pose", "recon_unposed"],
        help="Evaluation modes",
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
        "--eval_only",
        action="store_true",
        help="Only run evaluation (skip inference)",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Only print saved metrics",
    )
    args = parser.parse_args()

    # Create evaluator
    evaluator = Evaluator(
        work_dir=args.work_dir,
        datas=args.datasets,
        modes=args.modes,
        max_frames=args.max_frames,
        scenes=args.scenes,
    )

    if args.print_only:
        evaluator.print_metrics()
        return

    if args.eval_only:
        metrics = evaluator.eval()
        evaluator.print_metrics(metrics)
        return

    # Load LoRA model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api = LoRADepthAnything3(
        base_model=args.base_model,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    ).to(device)

    # Run inference and evaluation
    evaluator.infer(api)
    metrics = evaluator.eval()
    evaluator.print_metrics(metrics)


if __name__ == "__main__":
    main()
