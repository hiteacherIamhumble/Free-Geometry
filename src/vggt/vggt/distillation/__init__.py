"""
VGGT Knowledge Distillation Module.

This module provides components for distilling knowledge from a teacher VGGT model
(8 views, frozen) to a student VGGT model (4 views, LoRA-adapted).
"""

from .models import (
    VGGTDistillationOutput,
    VGGTTeacherModel,
    VGGTStudentModel,
    count_parameters,
)
from .dataset import VGGTDistillDataset

__all__ = [
    "VGGTDistillationOutput",
    "VGGTTeacherModel",
    "VGGTStudentModel",
    "VGGTDistillDataset",
    "count_parameters",
]
