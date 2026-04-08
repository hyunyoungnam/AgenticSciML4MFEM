"""
Evaluation pipeline for MFEM mesh validation and scoring.

Coordinates mesh validation, solver execution, and post-simulation
metric extraction.
"""

from piano.evaluation.pipeline import (
    EvaluationPipeline,
    EvaluationResult,
    PreflightResult,
    PreflightStatus,
)
from piano.evaluation.metrics import MetricsCalculator, MeshQualityMetrics

__all__ = [
    "EvaluationPipeline",
    "EvaluationResult",
    "PreflightResult",
    "PreflightStatus",
    "MetricsCalculator",
    "MeshQualityMetrics",
]
