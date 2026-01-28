"""
Token Inspection Tool for DA3 Knowledge Distillation.

This module provides utilities to:
1. Save teacher and student tokens (before DPT head) to files
2. Load and compare tokens with various analysis modes
3. Visualize feature distributions and differences

Usage:
    # Save tokens during training
    from depth_anything_3.distillation.inspect import save_distillation_tokens
    save_distillation_tokens(teacher_output, student_output, './tokens', step=100)

    # Analyze saved tokens
    python -m depth_anything_3.distillation.inspect tokens_step100.pt --plot
"""

import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from depth_anything_3.distillation.models import DistillationOutput


# =============================================================================
# Token Saving/Loading Functions
# =============================================================================

def save_distillation_tokens(
    teacher_output: DistillationOutput,
    student_output: DistillationOutput,
    save_dir: str,
    prefix: str = "tokens",
    step: int = 0,
    student_frame_indices: Optional[List[int]] = None,
) -> str:
    """
    Save teacher and student tokens to a single .pt file.

    Args:
        teacher_output: DistillationOutput from teacher (8 views)
        student_output: DistillationOutput from student (4 views)
        save_dir: Directory to save tokens
        prefix: Filename prefix
        step: Training step number
        student_frame_indices: Which teacher frames correspond to student views

    Returns:
        Path to saved file
    """
    os.makedirs(save_dir, exist_ok=True)

    # Default student frame indices
    if student_frame_indices is None:
        student_frame_indices = [0, 2, 4, 6]

    # Prepare data - detach and move to CPU
    def prepare_dict(d: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in d.items()}

    data = {
        'teacher': {
            'layer_features': prepare_dict(teacher_output.layer_features),
            'local_features': prepare_dict(teacher_output.local_features),
            'global_features': prepare_dict(teacher_output.global_features),
            'camera_tokens': prepare_dict(teacher_output.camera_tokens),
        },
        'student': {
            'layer_features': prepare_dict(student_output.layer_features),
            'local_features': prepare_dict(student_output.local_features),
            'global_features': prepare_dict(student_output.global_features),
            'camera_tokens': prepare_dict(student_output.camera_tokens),
        },
        'metadata': {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'output_layers': list(teacher_output.layer_features.keys()),
            'student_frame_indices': student_frame_indices,
        }
    }

    # Save
    filename = f"{prefix}_step{step}.pt"
    filepath = os.path.join(save_dir, filename)
    torch.save(data, filepath)
    print(f"Saved distillation tokens to {filepath}")

    return filepath


def load_distillation_tokens(path: str) -> Dict:
    """
    Load saved tokens from file.

    Args:
        path: Path to .pt file

    Returns:
        Dictionary with 'teacher', 'student', and 'metadata' keys
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Token file not found: {path}")

    data = torch.load(path, map_location='cpu')
    print(f"Loaded tokens from {path}")
    print(f"  Step: {data['metadata']['step']}")
    print(f"  Layers: {data['metadata']['output_layers']}")
    print(f"  Timestamp: {data['metadata']['timestamp']}")

    return data


# =============================================================================
# TokenInspector Class
# =============================================================================

class TokenInspector:
    """
    Analyze and compare teacher/student tokens.

    Provides methods for:
    - Computing statistics (mean, std, norm, etc.)
    - Comparing features (cosine similarity, MSE, L2 distance)
    - Visualizing distributions and differences
    """

    # Map feature type names to dict keys
    FEATURE_MAP = {
        'global': 'global_features',
        'local': 'local_features',
        'camera': 'camera_tokens',
        'full': 'layer_features',
    }

    def __init__(
        self,
        tokens: Dict,
        student_frame_indices: Optional[List[int]] = None,
    ):
        """
        Initialize TokenInspector.

        Args:
            tokens: Dictionary from load_distillation_tokens()
            student_frame_indices: Which teacher frames correspond to student
        """
        self.teacher = tokens['teacher']
        self.student = tokens['student']
        self.metadata = tokens['metadata']
        self.output_layers = self.metadata['output_layers']

        # Get student frame indices from metadata or default
        self.student_frame_indices = (
            student_frame_indices or
            self.metadata.get('student_frame_indices', [0, 2, 4, 6])
        )

    def _get_feature_key(self, feature_type: str) -> str:
        """Convert feature type name to dict key."""
        return self.FEATURE_MAP.get(feature_type, feature_type)

    def _get_shapes(self, layer_idx: int, feature_type: str) -> Tuple[tuple, tuple]:
        """Get tensor shapes for teacher and student."""
        key = self._get_feature_key(feature_type)
        teacher_shape = self.teacher[key][layer_idx].shape
        student_shape = self.student[key][layer_idx].shape
        return teacher_shape, student_shape

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def compute_statistics(
        self,
        feature_type: str = 'global',
        layer_idx: Optional[int] = None,
    ) -> Dict:
        """
        Compute statistics for teacher and student features.

        Args:
            feature_type: 'global', 'local', 'camera', or 'full'
            layer_idx: Specific layer index (None for all layers)

        Returns:
            Dictionary with statistics for each layer
        """
        key = self._get_feature_key(feature_type)
        layers = [layer_idx] if layer_idx else self.output_layers

        results = {}
        for layer in layers:
            teacher_feat = self.teacher[key][layer].float()
            student_feat = self.student[key][layer].float()

            results[layer] = {
                'teacher': self._compute_tensor_stats(teacher_feat),
                'student': self._compute_tensor_stats(student_feat),
            }

        return results

    def _compute_tensor_stats(self, tensor: torch.Tensor) -> Dict:
        """Compute statistics for a single tensor."""
        # Flatten all but last dimension for per-feature stats
        flat = tensor.reshape(-1, tensor.shape[-1])

        return {
            'shape': list(tensor.shape),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'l2_norm_mean': torch.norm(flat, p=2, dim=-1).mean().item(),
            'l2_norm_std': torch.norm(flat, p=2, dim=-1).std().item(),
        }

    # -------------------------------------------------------------------------
    # Comparisons
    # -------------------------------------------------------------------------

    def compute_cosine_similarity(
        self,
        layer_idx: int,
        feature_type: str = 'global',
    ) -> torch.Tensor:
        """
        Compute per-patch cosine similarity between matched teacher/student views.

        Args:
            layer_idx: Layer index to compare
            feature_type: 'global', 'local', or 'full'

        Returns:
            Tensor of cosine similarities [B, num_student_views, P]
        """
        key = self._get_feature_key(feature_type)
        teacher_feat = self.teacher[key][layer_idx].float()  # [B, 8, P, C]
        student_feat = self.student[key][layer_idx].float()  # [B, 4, P, C]

        # Select matching teacher frames
        teacher_selected = teacher_feat[:, self.student_frame_indices, :, :]  # [B, 4, P, C]

        # Compute cosine similarity along feature dimension
        teacher_norm = F.normalize(teacher_selected, p=2, dim=-1)
        student_norm = F.normalize(student_feat, p=2, dim=-1)
        cos_sim = (teacher_norm * student_norm).sum(dim=-1)  # [B, 4, P]

        return cos_sim

    def compute_mse(
        self,
        layer_idx: int,
        feature_type: str = 'global',
    ) -> torch.Tensor:
        """
        Compute per-patch MSE between matched teacher/student views.

        Args:
            layer_idx: Layer index to compare
            feature_type: 'global', 'local', or 'full'

        Returns:
            Tensor of MSE values [B, num_student_views, P]
        """
        key = self._get_feature_key(feature_type)
        teacher_feat = self.teacher[key][layer_idx].float()  # [B, 8, P, C]
        student_feat = self.student[key][layer_idx].float()  # [B, 4, P, C]

        # Select matching teacher frames
        teacher_selected = teacher_feat[:, self.student_frame_indices, :, :]  # [B, 4, P, C]

        # Compute MSE along feature dimension
        mse = ((teacher_selected - student_feat) ** 2).mean(dim=-1)  # [B, 4, P]

        return mse

    def compute_feature_distance(
        self,
        layer_idx: int,
        feature_type: str = 'global',
    ) -> Dict:
        """
        Compute L2 distance statistics between teacher and student.

        Args:
            layer_idx: Layer index to compare
            feature_type: 'global', 'local', or 'full'

        Returns:
            Dictionary with distance statistics
        """
        key = self._get_feature_key(feature_type)
        teacher_feat = self.teacher[key][layer_idx].float()
        student_feat = self.student[key][layer_idx].float()

        # Select matching teacher frames
        teacher_selected = teacher_feat[:, self.student_frame_indices, :, :]

        # Compute L2 distance per patch
        diff = teacher_selected - student_feat
        l2_dist = torch.norm(diff, p=2, dim=-1)  # [B, 4, P]

        return {
            'l2_mean': l2_dist.mean().item(),
            'l2_std': l2_dist.std().item(),
            'l2_min': l2_dist.min().item(),
            'l2_max': l2_dist.max().item(),
            'l2_per_view_mean': l2_dist.mean(dim=(0, 2)).tolist(),
        }

    def compute_camera_token_similarity(self, layer_idx: int) -> Dict:
        """
        Compute similarity between teacher and student camera tokens.

        Args:
            layer_idx: Layer index to compare

        Returns:
            Dictionary with similarity metrics
        """
        teacher_cam = self.teacher['camera_tokens'][layer_idx].float()  # [B, 8, C]
        student_cam = self.student['camera_tokens'][layer_idx].float()  # [B, 4, C]

        # Select matching teacher frames
        teacher_selected = teacher_cam[:, self.student_frame_indices, :]  # [B, 4, C]

        # Cosine similarity
        teacher_norm = F.normalize(teacher_selected, p=2, dim=-1)
        student_norm = F.normalize(student_cam, p=2, dim=-1)
        cos_sim = (teacher_norm * student_norm).sum(dim=-1)  # [B, 4]

        # L2 distance
        l2_dist = torch.norm(teacher_selected - student_cam, p=2, dim=-1)

        return {
            'cosine_sim_mean': cos_sim.mean().item(),
            'cosine_sim_per_view': cos_sim.mean(dim=0).tolist(),
            'l2_dist_mean': l2_dist.mean().item(),
            'l2_dist_per_view': l2_dist.mean(dim=0).tolist(),
        }

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------

    def plot_feature_distributions(
        self,
        layer_idx: int,
        feature_type: str = 'global',
        save_path: Optional[str] = None,
    ):
        """
        Plot histogram of feature magnitudes for teacher vs student.

        Args:
            layer_idx: Layer index to plot
            feature_type: 'global', 'local', or 'full'
            save_path: Path to save figure (None for display)
        """
        import matplotlib.pyplot as plt

        key = self._get_feature_key(feature_type)
        teacher_feat = self.teacher[key][layer_idx].float()
        student_feat = self.student[key][layer_idx].float()

        # Select matching teacher frames
        teacher_selected = teacher_feat[:, self.student_frame_indices, :, :]

        # Compute L2 norms
        teacher_norms = torch.norm(teacher_selected, p=2, dim=-1).flatten().numpy()
        student_norms = torch.norm(student_feat, p=2, dim=-1).flatten().numpy()

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # L2 norm distribution
        axes[0].hist(teacher_norms, bins=50, alpha=0.6, label='Teacher', color='blue')
        axes[0].hist(student_norms, bins=50, alpha=0.6, label='Student', color='orange')
        axes[0].set_xlabel('L2 Norm')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Feature L2 Norm Distribution (Layer {layer_idx})')
        axes[0].legend()

        # Mean feature values
        teacher_means = teacher_selected.mean(dim=-1).flatten().numpy()
        student_means = student_feat.mean(dim=-1).flatten().numpy()

        axes[1].hist(teacher_means, bins=50, alpha=0.6, label='Teacher', color='blue')
        axes[1].hist(student_means, bins=50, alpha=0.6, label='Student', color='orange')
        axes[1].set_xlabel('Mean Feature Value')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Mean Feature Distribution (Layer {layer_idx})')
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved feature distribution plot to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_cosine_similarity_heatmap(
        self,
        layer_idx: int,
        view_idx: int = 0,
        batch_idx: int = 0,
        feature_type: str = 'global',
        save_path: Optional[str] = None,
    ):
        """
        Plot 2D heatmap of per-patch cosine similarities.

        Args:
            layer_idx: Layer index to plot
            view_idx: View index (0-3 for student views)
            batch_idx: Batch index
            feature_type: 'global', 'local', or 'full'
            save_path: Path to save figure (None for display)
        """
        import matplotlib.pyplot as plt

        cos_sim = self.compute_cosine_similarity(layer_idx, feature_type)  # [B, 4, P]

        # Get cosine similarity for specific batch and view
        sim_flat = cos_sim[batch_idx, view_idx, :]  # [P]

        # Determine spatial dimensions (assume square)
        P = sim_flat.shape[0]
        H = W = int(P ** 0.5)
        if H * W != P:
            print(f"Warning: P={P} is not a perfect square, truncating to {H*H}")
            sim_flat = sim_flat[:H*H]

        sim_2d = sim_flat.reshape(H, W).numpy()

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(sim_2d, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f'Cosine Similarity (Layer {layer_idx}, View {view_idx})\nMean: {sim_2d.mean():.4f}')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved cosine similarity heatmap to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_difference_map(
        self,
        layer_idx: int,
        view_idx: int = 0,
        batch_idx: int = 0,
        feature_type: str = 'global',
        save_path: Optional[str] = None,
    ):
        """
        Plot 2D heatmap of feature L2 distances.

        Args:
            layer_idx: Layer index to plot
            view_idx: View index (0-3 for student views)
            batch_idx: Batch index
            feature_type: 'global', 'local', or 'full'
            save_path: Path to save figure (None for display)
        """
        import matplotlib.pyplot as plt

        key = self._get_feature_key(feature_type)
        teacher_feat = self.teacher[key][layer_idx].float()
        student_feat = self.student[key][layer_idx].float()

        # Select matching teacher frame
        teacher_idx = self.student_frame_indices[view_idx]
        teacher_view = teacher_feat[batch_idx, teacher_idx, :, :]  # [P, C]
        student_view = student_feat[batch_idx, view_idx, :, :]  # [P, C]

        # Compute L2 distance per patch
        l2_dist = torch.norm(teacher_view - student_view, p=2, dim=-1)  # [P]

        # Determine spatial dimensions
        P = l2_dist.shape[0]
        H = W = int(P ** 0.5)
        if H * W != P:
            print(f"Warning: P={P} is not a perfect square, truncating to {H*H}")
            l2_dist = l2_dist[:H*H]

        dist_2d = l2_dist.reshape(H, W).numpy()

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(dist_2d, cmap='hot')
        ax.set_title(f'L2 Distance (Layer {layer_idx}, View {view_idx})\nMean: {dist_2d.mean():.4f}')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        plt.colorbar(im, ax=ax, label='L2 Distance')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved difference map to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_layer_comparison(
        self,
        feature_type: str = 'global',
        save_path: Optional[str] = None,
    ):
        """
        Plot comparison metrics across all layers.

        Args:
            feature_type: 'global', 'local', or 'full'
            save_path: Path to save figure (None for display)
        """
        import matplotlib.pyplot as plt

        layers = self.output_layers
        cos_sims = []
        l2_dists = []

        for layer in layers:
            cos_sim = self.compute_cosine_similarity(layer, feature_type)
            cos_sims.append(cos_sim.mean().item())

            dist = self.compute_feature_distance(layer, feature_type)
            l2_dists.append(dist['l2_mean'])

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Cosine similarity per layer
        axes[0].bar(range(len(layers)), cos_sims, color='green', alpha=0.7)
        axes[0].set_xticks(range(len(layers)))
        axes[0].set_xticklabels([str(l) for l in layers])
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Cosine Similarity')
        axes[0].set_title('Mean Cosine Similarity per Layer')
        axes[0].set_ylim(0, 1)

        # L2 distance per layer
        axes[1].bar(range(len(layers)), l2_dists, color='red', alpha=0.7)
        axes[1].set_xticks(range(len(layers)))
        axes[1].set_xticklabels([str(l) for l in layers])
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('L2 Distance')
        axes[1].set_title('Mean L2 Distance per Layer')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved layer comparison plot to {save_path}")
            plt.close()
        else:
            plt.show()

    # -------------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        """
        Generate summary of all statistics.

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("DISTILLATION TOKEN INSPECTION SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Step: {self.metadata['step']}")
        lines.append(f"Timestamp: {self.metadata['timestamp']}")
        lines.append(f"Output Layers: {self.output_layers}")
        lines.append(f"Student Frame Indices: {self.student_frame_indices}")
        lines.append("")

        for layer in self.output_layers:
            lines.append("-" * 70)
            lines.append(f"LAYER {layer}")
            lines.append("-" * 70)

            # Global features
            lines.append("\nGlobal Features:")
            stats = self.compute_statistics('global', layer)[layer]
            t_shape, s_shape = self._get_shapes(layer, 'global')
            lines.append(f"  Teacher shape: {list(t_shape)}")
            lines.append(f"  Student shape: {list(s_shape)}")
            lines.append(f"  Teacher - mean: {stats['teacher']['mean']:.6f}, std: {stats['teacher']['std']:.6f}, L2 norm: {stats['teacher']['l2_norm_mean']:.4f}")
            lines.append(f"  Student - mean: {stats['student']['mean']:.6f}, std: {stats['student']['std']:.6f}, L2 norm: {stats['student']['l2_norm_mean']:.4f}")

            # Comparison metrics
            cos_sim = self.compute_cosine_similarity(layer, 'global')
            mse = self.compute_mse(layer, 'global')
            dist = self.compute_feature_distance(layer, 'global')

            lines.append(f"\n  Comparison (Global):")
            lines.append(f"    Cosine Similarity: {cos_sim.mean():.6f} (std: {cos_sim.std():.6f})")
            lines.append(f"    MSE: {mse.mean():.6f} (std: {mse.std():.6f})")
            lines.append(f"    L2 Distance: {dist['l2_mean']:.6f} (std: {dist['l2_std']:.6f})")

            # Camera tokens
            cam_sim = self.compute_camera_token_similarity(layer)
            lines.append(f"\n  Camera Tokens:")
            lines.append(f"    Cosine Similarity: {cam_sim['cosine_sim_mean']:.6f}")
            lines.append(f"    Per-view: {[f'{v:.4f}' for v in cam_sim['cosine_sim_per_view']]}")
            lines.append(f"    L2 Distance: {cam_sim['l2_dist_mean']:.6f}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def compare_layers(self) -> Dict:
        """
        Compare statistics across different layers.

        Returns:
            Dictionary with per-layer comparison metrics
        """
        results = {}
        for layer in self.output_layers:
            cos_sim = self.compute_cosine_similarity(layer, 'global')
            mse = self.compute_mse(layer, 'global')
            dist = self.compute_feature_distance(layer, 'global')
            cam_sim = self.compute_camera_token_similarity(layer)

            results[layer] = {
                'global_cosine_sim': cos_sim.mean().item(),
                'global_mse': mse.mean().item(),
                'global_l2_dist': dist['l2_mean'],
                'camera_cosine_sim': cam_sim['cosine_sim_mean'],
                'camera_l2_dist': cam_sim['l2_dist_mean'],
            }

        return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Inspect saved distillation tokens',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic inspection
    python -m depth_anything_3.distillation.inspect tokens_step100.pt

    # With plots
    python -m depth_anything_3.distillation.inspect tokens_step100.pt --plot

    # Specific layer and feature type
    python -m depth_anything_3.distillation.inspect tokens_step100.pt --layer 39 --feature global --plot
        """
    )
    parser.add_argument('token_file', help='Path to saved tokens .pt file')
    parser.add_argument('--layer', type=int, default=None,
                        help='Specific layer index to analyze (default: all)')
    parser.add_argument('--feature', choices=['global', 'local', 'camera', 'full'],
                        default='global', help='Feature type to analyze')
    parser.add_argument('--output_dir', default='./inspect_output',
                        help='Output directory for plots')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--view', type=int, default=0,
                        help='View index for heatmaps (0-3)')
    parser.add_argument('--batch', type=int, default=0,
                        help='Batch index for heatmaps')

    args = parser.parse_args()

    # Load tokens
    tokens = load_distillation_tokens(args.token_file)

    # Create inspector
    inspector = TokenInspector(tokens)

    # Print summary
    print(inspector.summary())

    # Generate plots if requested
    if args.plot:
        os.makedirs(args.output_dir, exist_ok=True)

        layers = [args.layer] if args.layer else inspector.output_layers

        for layer in layers:
            # Feature distribution
            inspector.plot_feature_distributions(
                layer,
                args.feature,
                save_path=os.path.join(args.output_dir, f'dist_layer{layer}_{args.feature}.png')
            )

            # Cosine similarity heatmap
            if args.feature != 'camera':
                inspector.plot_cosine_similarity_heatmap(
                    layer,
                    view_idx=args.view,
                    batch_idx=args.batch,
                    feature_type=args.feature,
                    save_path=os.path.join(args.output_dir, f'cos_sim_layer{layer}_view{args.view}.png')
                )

                # Difference map
                inspector.plot_difference_map(
                    layer,
                    view_idx=args.view,
                    batch_idx=args.batch,
                    feature_type=args.feature,
                    save_path=os.path.join(args.output_dir, f'diff_layer{layer}_view{args.view}.png')
                )

        # Layer comparison
        if len(layers) > 1:
            inspector.plot_layer_comparison(
                args.feature,
                save_path=os.path.join(args.output_dir, f'layer_comparison_{args.feature}.png')
            )

        print(f"\nPlots saved to {args.output_dir}")


if __name__ == '__main__':
    main()
