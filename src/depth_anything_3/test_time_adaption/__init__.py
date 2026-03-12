"""
Free-Geometry module for Depth Anything 3.

This module provides the active DA3 Free-Geometry components.
"""

from depth_anything_3.test_time_adaption.config import FreeGeometryConfig
from depth_anything_3.test_time_adaption.dataset import ScanNetPPFreeGeometryDataset
from depth_anything_3.test_time_adaption.benchmark_dataset import BenchmarkFreeGeometryDataset
from depth_anything_3.test_time_adaption.losses import (
    PatchHuberCosineLoss,
    DA3CrossFrameCFAngleLoss,
    DA3CrossFrameCFDistanceLoss,
)
from depth_anything_3.test_time_adaption.models import (
    TeacherModel,
    StudentModel,
    DA3StudentFinetune,
    FreeGeometryOutput,
)

__all__ = [
    # Config
    'FreeGeometryConfig',
    # Dataset
    'ScanNetPPFreeGeometryDataset',
    'BenchmarkFreeGeometryDataset',
    # Losses
    'PatchHuberCosineLoss',
    'DA3CrossFrameCFAngleLoss',
    'DA3CrossFrameCFDistanceLoss',
    # Models
    'TeacherModel',
    'StudentModel',
    'DA3StudentFinetune',
    'FreeGeometryOutput',
]
