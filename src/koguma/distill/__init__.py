"""Knowledge distillation components for Koguma-LM."""

from .distiller import MultiTeacherDistiller
from .data_generator import DistillationDataGenerator

__all__ = ["MultiTeacherDistiller", "DistillationDataGenerator"]