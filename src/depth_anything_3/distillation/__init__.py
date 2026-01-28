"""
Distillation module for Depth Anything 3.

This module provides components for knowledge distillation training:
- Dataset: ScanNet++ dataset for multi-view distillation
- Models: Teacher and student model wrappers (using HuggingFace PEFT for LoRA)
- Losses: Distillation loss functions
- Config: Training configuration
- Inspect: Token inspection and comparison tools
"""

from depth_anything_3.distillation.config import DistillConfig
from depth_anything_3.distillation.dataset import ScanNetPPDistillDataset
from depth_anything_3.distillation.losses import (
    DA3DistillationLoss,
    RobustRegressionLoss,
    CosineDistillLoss,
    CameraTokenOnlyLoss,
    CameraTokenMSELoss,
    CameraTokenCosineLoss,
)
from depth_anything_3.distillation.models import (
    TeacherModel,
    StudentModel,
    DistillationOutput,
)
from depth_anything_3.distillation.inspect import (
    save_distillation_tokens,
    load_distillation_tokens,
    TokenInspector,
)

__all__ = [
    # Config
    'DistillConfig',
    # Dataset
    'ScanNetPPDistillDataset',
    # Losses
    'DA3DistillationLoss',
    'RobustRegressionLoss',
    'CosineDistillLoss',
    'CameraTokenOnlyLoss',
    'CameraTokenMSELoss',
    'CameraTokenCosineLoss',
    # Models
    'TeacherModel',
    'StudentModel',
    'DistillationOutput',
    # Inspect
    'save_distillation_tokens',
    'load_distillation_tokens',
    'TokenInspector',
]
