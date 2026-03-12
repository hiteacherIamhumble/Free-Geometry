"""
Training configuration for DA3 Free-Geometry.

This module provides a dataclass-based configuration for Free-Geometry training,
including model, data, training, and loss hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    # Model name (HuggingFace or local path)
    model_name: str = "depth-anything/DA3-GIANT-1.1"

    # Output layers for feature extraction (must match DPT input layers for Giant: 19, 27, 33, 39)
    output_layers: List[int] = field(default_factory=lambda: [19, 27, 33, 39])

    # Embedding dimension (1536 for Giant)
    embed_dim: int = 1536

    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1  # Add dropout for regularization

    # Whether to train camera token
    train_camera_token: bool = True  # Enable to adapt pose estimation for 4 views


@dataclass
class DataConfig:
    """Data configuration."""
    # Data root directory
    data_root: str = "./data"

    # Number of views for teacher (8 or 16)
    num_views: int = 8

    # Student frame indices (which teacher frames to use for student)
    # For 8-view teacher: [0, 2, 4, 6]
    # For 16-view teacher: [0, 4, 8, 12]
    student_indices: List[int] = field(default_factory=lambda: [0, 2, 4, 6])

    # Image size (H, W)
    image_size: tuple = (518, 518)

    # Data augmentation
    augment: bool = True

    # Load camera parameters
    load_cameras: bool = False

    # DataLoader settings
    batch_size: int = 2
    num_workers: int = 4


@dataclass
class LossConfig:
    """Loss configuration."""
    # Feature loss weights
    robust_weight: float = 1.0
    cosine_weight: float = 1.0
    kl_weight: float = 1.0  # Re-enabled for pose preservation

    # Camera token loss weight
    # NOTE: Camera tokens diverge significantly between 8-view teacher and 4-view student
    # Use lower weight to prevent dominating the loss
    camera_token_weight: float = 0.5  # Reduced from 2.0 to balance with feature loss

    # Robust loss parameters
    robust_alpha: float = 0.5
    robust_scaling_c: float = 0.5  # Increased from 0.25 to reduce loss magnitude

    # Cosine loss temperature
    cosine_temperature: float = 0.1

    # KL loss temperature
    kl_temperature: float = 0.07

    # Layers for each loss type (match DPT input layers)
    robust_cosine_layers: List[int] = field(default_factory=lambda: [19, 27, 33, 39])  # All DPT layers
    kl_layers: List[int] = field(default_factory=lambda: [33, 39])  # KL only at later layers (cross-view)

    # Camera token only mode - disables all other losses
    # Uses only layer 39 with full 3072-dim token (what camera decoder receives)
    camera_token_only: bool = False  # When True, only use camera token loss

    # Camera token loss type (only used when camera_token_only=True)
    # Options: "smooth_l1_cosine" (default), "mse", "cosine"
    camera_token_loss_type: str = "cosine"  # Use pure cosine similarity by default


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Number of epochs
    epochs: int = 10

    # Learning rate
    lr: float = 1e-4

    # Weight decay
    weight_decay: float = 0.01

    # Learning rate scheduler
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant"
    warmup_steps: int = 500

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Mixed precision training
    use_amp: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000

    # Checkpointing
    output_dir: str = "./checkpoints/tta"
    save_total_limit: int = 3

    # Random seed
    seed: int = 42

    # Device
    device: str = "cuda"

    # Gradient accumulation
    gradient_accumulation_steps: int = 1


@dataclass
class FreeGeometryConfig:
    """Complete Free-Geometry configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "FreeGeometryConfig":
        """Create config from dictionary."""
        model_cfg = ModelConfig(**config_dict.get('model', {}))
        data_cfg = DataConfig(**config_dict.get('data', {}))
        loss_cfg = LossConfig(**config_dict.get('loss', {}))
        training_cfg = TrainingConfig(**config_dict.get('training', {}))

        return cls(
            model=model_cfg,
            data=data_cfg,
            loss=loss_cfg,
            training=training_cfg,
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)


def get_default_config() -> FreeGeometryConfig:
    """Get default Free-Geometry configuration."""
    return FreeGeometryConfig()


def get_debug_config() -> FreeGeometryConfig:
    """Get configuration for debugging (smaller batch, fewer steps)."""
    config = FreeGeometryConfig()
    config.data.batch_size = 1
    config.data.num_workers = 0
    config.training.epochs = 1
    config.training.log_interval = 1
    config.training.eval_interval = 10
    config.training.save_interval = 10
    return config
