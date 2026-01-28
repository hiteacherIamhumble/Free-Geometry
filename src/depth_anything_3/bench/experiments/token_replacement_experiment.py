"""
Token Replacement Experiment for Local Feature Analysis.

This experiment examines whether high-MSE local tokens are outliers hurting
performance or important features by:
1. Running teacher with 8 views, extracting tokens for frames [0,2,4,6]
2. Running student with 4 views (frames [0,2,4,6])
3. Computing MSE between local tokens
4. Replacing top-10% or bottom-90% MSE tokens with teacher tokens
5. Decoding and comparing results

Usage:
    python token_replacement_experiment.py --data_dir /path/to/data --output_dir /path/to/output
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3


def parse_args():
    parser = argparse.ArgumentParser(description="Token replacement experiment")
    parser.add_argument(
        "--model_name",
        type=str,
        default="depth-anything/DA3-GIANT-1.1",
        help="Model name or path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing image sequences",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./token_replacement_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=10,
        help="Number of sequences to process",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=518,
        help="Image size for processing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--replacement_percentiles",
        type=str,
        default="10,20,30,50",
        help="Comma-separated percentiles for replacement experiments",
    )
    return parser.parse_args()


class TokenReplacementExperiment:
    """
    Experiment to analyze the impact of high-MSE local tokens.
    """

    STUDENT_FRAME_INDICES = [0, 2, 4, 6]  # Which teacher frames map to student
    EMBED_DIM = 1536  # Giant model embedding dimension

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        image_size: int = 518,
    ):
        self.device = device
        self.image_size = image_size

        # Load model
        print(f"Loading model: {model_name}")
        self.model = DepthAnything3.from_pretrained(model_name).to(device)
        self.model.eval()

        # Get the underlying network
        self.net = self.model.model

    def load_sequence(self, image_paths: List[str]) -> torch.Tensor:
        """Load and preprocess a sequence of images."""
        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = img.resize((self.image_size, self.image_size))
            img = torch.from_numpy(np.array(img)).float() / 255.0
            img = img.permute(2, 0, 1)  # HWC -> CHW
            images.append(img)

        # Stack: [S, 3, H, W] -> [1, S, 3, H, W]
        images = torch.stack(images, dim=0).unsqueeze(0)
        return images.to(self.device)

    @torch.no_grad()
    def run_teacher_8view(
        self, images_8view: torch.Tensor
    ) -> Tuple[List, int, int, torch.Tensor, torch.Tensor]:
        """
        Run teacher with 8 views and extract features.

        Returns:
            feats: List of (features, camera_token) tuples
            H, W: Image dimensions
            local_feats: Local features [B, 8, P, 1536]
            global_feats: Global features [B, 8, P, 1536]
        """
        feats, aux_feats, H, W = self.net.forward_backbone_only(
            images_8view, ref_view_strategy="first"
        )

        # Extract local and global from last layer (layer 39)
        # feats[-1] = (features [B, S, P, 3072], camera_token [B, S, 3072])
        full_feats = feats[-1][0]  # [B, 8, P, 3072]
        local_feats = full_feats[..., :self.EMBED_DIM]  # [B, 8, P, 1536]
        global_feats = full_feats[..., self.EMBED_DIM:]  # [B, 8, P, 1536]

        return feats, H, W, local_feats, global_feats

    @torch.no_grad()
    def run_student_4view(
        self, images_4view: torch.Tensor
    ) -> Tuple[List, int, int, torch.Tensor, torch.Tensor]:
        """
        Run student with 4 views and extract features.

        Returns:
            feats: List of (features, camera_token) tuples
            H, W: Image dimensions
            local_feats: Local features [B, 4, P, 1536]
            global_feats: Global features [B, 4, P, 1536]
        """
        feats, aux_feats, H, W = self.net.forward_backbone_only(
            images_4view, ref_view_strategy="first"
        )

        # Extract local and global from last layer
        full_feats = feats[-1][0]  # [B, 4, P, 3072]
        local_feats = full_feats[..., :self.EMBED_DIM]  # [B, 4, P, 1536]
        global_feats = full_feats[..., self.EMBED_DIM:]  # [B, 4, P, 1536]

        return feats, H, W, local_feats, global_feats

    def compute_token_mse(
        self,
        teacher_local: torch.Tensor,
        student_local: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token MSE between teacher and student local features.

        Args:
            teacher_local: [B, 4, P, 1536] teacher local features (selected frames)
            student_local: [B, 4, P, 1536] student local features

        Returns:
            mse_per_token: [B, 4, P] MSE for each token
            cos_sim_per_token: [B, 4, P] Cosine similarity for each token
        """
        # MSE per token (mean over feature dim)
        mse_per_token = ((teacher_local - student_local) ** 2).mean(dim=-1)  # [B, 4, P]

        # Cosine similarity per token
        cos_sim = F.cosine_similarity(teacher_local, student_local, dim=-1)  # [B, 4, P]

        return mse_per_token, cos_sim

    def replace_tokens_by_percentile(
        self,
        student_feats: List,
        teacher_local: torch.Tensor,
        student_local: torch.Tensor,
        mse_per_token: torch.Tensor,
        percentile: float,
        replace_high: bool = True,
    ) -> List:
        """
        Replace student local tokens based on MSE percentile.

        Args:
            student_feats: Original student feats list
            teacher_local: [B, 4, P, 1536] teacher local features
            student_local: [B, 4, P, 1536] student local features
            mse_per_token: [B, 4, P] MSE per token
            percentile: Percentile threshold (e.g., 10 for top 10%)
            replace_high: If True, replace high-MSE tokens; else replace low-MSE

        Returns:
            Modified feats list with replaced tokens
        """
        B, S, P, C = student_local.shape

        # Flatten MSE for percentile computation
        mse_flat = mse_per_token.reshape(-1)

        if replace_high:
            # Replace top percentile (high MSE tokens)
            threshold = torch.quantile(mse_flat, 1 - percentile / 100)
            mask = mse_per_token >= threshold  # [B, 4, P]
        else:
            # Replace bottom percentile (low MSE tokens)
            threshold = torch.quantile(mse_flat, percentile / 100)
            mask = mse_per_token <= threshold  # [B, 4, P]

        # Create modified local features
        modified_local = student_local.clone()
        modified_local[mask] = teacher_local[mask]

        return modified_local, mask

    def create_modified_feats(
        self,
        student_feats: List,
        modified_local: torch.Tensor,
        student_global: torch.Tensor,
    ) -> List:
        """
        Create modified feats list with replaced local tokens.

        Args:
            student_feats: Original student feats list
            modified_local: [B, 4, P, 1536] modified local features
            student_global: [B, 4, P, 1536] original student global features

        Returns:
            Modified feats list ready for decoding
        """
        # Concatenate modified local with original global
        modified_full = torch.cat([modified_local, student_global], dim=-1)

        # Create new feats list (copy structure, replace last layer features)
        modified_feats = []
        for i, (feat, cam_token) in enumerate(student_feats):
            if i == len(student_feats) - 1:
                # Last layer: use modified features
                # Also need to modify camera token (first token)
                modified_cam = torch.cat([
                    modified_local[:, :, 0, :],  # local cls
                    student_global[:, :, 0, :]   # global cam token
                ], dim=-1)
                modified_feats.append((modified_full, modified_cam))
            else:
                modified_feats.append((feat, cam_token))

        return modified_feats

    @torch.no_grad()
    def decode_feats(self, feats: List, H: int, W: int) -> Dict:
        """Decode features to get depth and camera predictions."""
        output = self.net.forward_head_only(feats, H, W)
        return output

    @torch.no_grad()
    def run_single_experiment(
        self,
        images_8view: torch.Tensor,
        percentiles: List[float] = [10, 20, 30, 50],
    ) -> Dict:
        """
        Run token replacement experiment on a single sequence.

        Args:
            images_8view: [1, 8, 3, H, W] input images
            percentiles: List of percentiles to test

        Returns:
            Dict with results for each experiment variant
        """
        results = {}

        # 1. Run teacher with 8 views
        teacher_feats, H, W, teacher_local, teacher_global = self.run_teacher_8view(
            images_8view
        )

        # Select teacher frames matching student indices
        teacher_local_selected = teacher_local[:, self.STUDENT_FRAME_INDICES, :, :]
        teacher_global_selected = teacher_global[:, self.STUDENT_FRAME_INDICES, :, :]

        # 2. Run student with 4 views
        images_4view = images_8view[:, self.STUDENT_FRAME_INDICES, :, :, :]
        student_feats, _, _, student_local, student_global = self.run_student_4view(
            images_4view
        )

        # 3. Compute MSE and cosine similarity
        mse_per_token, cos_sim = self.compute_token_mse(
            teacher_local_selected, student_local
        )

        # Store statistics
        results["mse_stats"] = {
            "mean": mse_per_token.mean().item(),
            "std": mse_per_token.std().item(),
            "min": mse_per_token.min().item(),
            "max": mse_per_token.max().item(),
            "median": mse_per_token.median().item(),
        }
        results["cos_sim_stats"] = {
            "mean": cos_sim.mean().item(),
            "std": cos_sim.std().item(),
            "min": cos_sim.min().item(),
            "max": cos_sim.max().item(),
        }

        # 4. Baseline: decode original student features
        baseline_output = self.decode_feats(student_feats, H, W)
        results["baseline"] = {
            "depth": baseline_output.depth.cpu(),
            "extrinsics": baseline_output.extrinsics.cpu(),
            "intrinsics": baseline_output.intrinsics.cpu(),
        }

        # 5. Teacher baseline (8-view selected to 4)
        teacher_feats_4view = self._select_teacher_feats(teacher_feats)
        teacher_output = self.decode_feats(teacher_feats_4view, H, W)
        results["teacher_4view"] = {
            "depth": teacher_output.depth.cpu(),
            "extrinsics": teacher_output.extrinsics.cpu(),
            "intrinsics": teacher_output.intrinsics.cpu(),
        }

        return results, mse_per_token, cos_sim, student_feats, H, W, \
               teacher_local_selected, student_local, student_global

    def _select_teacher_feats(self, teacher_feats: List) -> List:
        """Select frames from teacher feats to match student indices."""
        selected_feats = []
        for feat, cam_token in teacher_feats:
            selected_feat = feat[:, self.STUDENT_FRAME_INDICES, :, :]
            selected_cam = cam_token[:, self.STUDENT_FRAME_INDICES, :]
            selected_feats.append((selected_feat, selected_cam))
        return selected_feats

    @torch.no_grad()
    def run_replacement_experiments(
        self,
        results: Dict,
        mse_per_token: torch.Tensor,
        student_feats: List,
        H: int,
        W: int,
        teacher_local: torch.Tensor,
        student_local: torch.Tensor,
        student_global: torch.Tensor,
        percentiles: List[float],
    ) -> Dict:
        """Run token replacement experiments for different percentiles."""

        for pct in percentiles:
            # Replace HIGH MSE tokens (outliers) with teacher
            modified_local_high, mask_high = self.replace_tokens_by_percentile(
                student_feats, teacher_local, student_local,
                mse_per_token, pct, replace_high=True
            )
            modified_feats_high = self.create_modified_feats(
                student_feats, modified_local_high, student_global
            )
            output_high = self.decode_feats(modified_feats_high, H, W)

            results[f"replace_high_{pct}pct"] = {
                "depth": output_high.depth.cpu(),
                "extrinsics": output_high.extrinsics.cpu(),
                "intrinsics": output_high.intrinsics.cpu(),
                "num_replaced": mask_high.sum().item(),
                "pct_replaced": mask_high.float().mean().item() * 100,
            }

            # Replace LOW MSE tokens with teacher
            modified_local_low, mask_low = self.replace_tokens_by_percentile(
                student_feats, teacher_local, student_local,
                mse_per_token, 100 - pct, replace_high=False
            )
            modified_feats_low = self.create_modified_feats(
                student_feats, modified_local_low, student_global
            )
            output_low = self.decode_feats(modified_feats_low, H, W)

            results[f"replace_low_{100-pct}pct"] = {
                "depth": output_low.depth.cpu(),
                "extrinsics": output_low.extrinsics.cpu(),
                "intrinsics": output_low.intrinsics.cpu(),
                "num_replaced": mask_low.sum().item(),
                "pct_replaced": mask_low.float().mean().item() * 100,
            }

        return results


def find_image_sequences(data_dir: str, num_sequences: int = 10) -> List[List[str]]:
    """Find image sequences in data directory."""
    data_path = Path(data_dir)
    sequences = []

    # Look for subdirectories with images
    for subdir in sorted(data_path.iterdir()):
        if not subdir.is_dir():
            continue

        # Find image files
        images = sorted(subdir.glob("*.jpg")) + sorted(subdir.glob("*.png"))
        if len(images) >= 8:
            sequences.append([str(img) for img in images[:8]])

        if len(sequences) >= num_sequences:
            break

    return sequences


def compute_depth_metrics(pred: torch.Tensor, ref: torch.Tensor) -> Dict:
    """Compute depth comparison metrics."""
    # Scale-invariant comparison
    pred_flat = pred.reshape(-1)
    ref_flat = ref.reshape(-1)

    # Compute scale factor
    scale = (ref_flat * pred_flat).sum() / (pred_flat * pred_flat).sum()
    pred_scaled = pred * scale

    # Compute metrics
    diff = (pred_scaled - ref).abs()
    rel_diff = diff / (ref + 1e-6)

    return {
        "mae": diff.mean().item(),
        "rmse": (diff ** 2).mean().sqrt().item(),
        "rel_mae": rel_diff.mean().item(),
        "scale": scale.item(),
    }


def main():
    args = parse_args()

    # Parse percentiles
    percentiles = [float(p) for p in args.replacement_percentiles.split(",")]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize experiment
    experiment = TokenReplacementExperiment(
        model_name=args.model_name,
        device=args.device,
        image_size=args.image_size,
    )

    # Find sequences
    sequences = find_image_sequences(args.data_dir, args.num_sequences)
    print(f"Found {len(sequences)} sequences")

    if len(sequences) == 0:
        print("No sequences found. Please check data_dir.")
        return

    # Aggregate results
    all_results = []

    for seq_idx, seq_paths in enumerate(tqdm(sequences, desc="Processing")):
        print(f"\nSequence {seq_idx + 1}/{len(sequences)}")

        # Load images
        images = experiment.load_sequence(seq_paths)

        # Run experiment
        results, mse, cos_sim, student_feats, H, W, \
            teacher_local, student_local, student_global = \
            experiment.run_single_experiment(images, percentiles)

        # Run replacement experiments
        results = experiment.run_replacement_experiments(
            results, mse, student_feats, H, W,
            teacher_local, student_local, student_global, percentiles
        )

        # Compute metrics comparing to teacher
        teacher_depth = results["teacher_4view"]["depth"]
        seq_metrics = {
            "seq_idx": seq_idx,
            "mse_stats": results["mse_stats"],
            "cos_sim_stats": results["cos_sim_stats"],
        }

        # Baseline vs teacher
        baseline_metrics = compute_depth_metrics(
            results["baseline"]["depth"], teacher_depth
        )
        seq_metrics["baseline_vs_teacher"] = baseline_metrics

        # Replacement experiments vs teacher
        for pct in percentiles:
            key_high = f"replace_high_{pct}pct"
            if key_high in results:
                metrics = compute_depth_metrics(
                    results[key_high]["depth"], teacher_depth
                )
                seq_metrics[f"{key_high}_vs_teacher"] = metrics

            key_low = f"replace_low_{100-pct}pct"
            if key_low in results:
                metrics = compute_depth_metrics(
                    results[key_low]["depth"], teacher_depth
                )
                seq_metrics[f"{key_low}_vs_teacher"] = metrics

        all_results.append(seq_metrics)

    # Aggregate and print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Average MSE stats
    avg_mse = np.mean([r["mse_stats"]["mean"] for r in all_results])
    avg_cos = np.mean([r["cos_sim_stats"]["mean"] for r in all_results])
    print(f"Avg local token MSE: {avg_mse:.4f}")
    print(f"Avg local token cos_sim: {avg_cos:.4f}")

    # Baseline metrics
    avg_baseline_mae = np.mean([
        r["baseline_vs_teacher"]["mae"] for r in all_results
    ])
    print(f"\nBaseline (student 4-view) vs Teacher:")
    print(f"  MAE: {avg_baseline_mae:.4f}")

    # Replacement experiment metrics
    print("\nReplacement Experiments vs Teacher:")
    for pct in percentiles:
        key_high = f"replace_high_{pct}pct_vs_teacher"
        key_low = f"replace_low_{100-pct}pct_vs_teacher"

        if key_high in all_results[0]:
            avg_mae = np.mean([r[key_high]["mae"] for r in all_results])
            print(f"  Replace HIGH {pct}% MSE tokens: MAE={avg_mae:.4f}")

        if key_low in all_results[0]:
            avg_mae = np.mean([r[key_low]["mae"] for r in all_results])
            print(f"  Replace LOW {100-pct}% MSE tokens: MAE={avg_mae:.4f}")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
