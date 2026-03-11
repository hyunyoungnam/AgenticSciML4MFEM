"""
Evaluation pipeline for Abaqus model validation and scoring.

Coordinates pre-flight checks, solver execution, and post-simulation
metric extraction.
"""

from meshforge.evaluation.pipeline import EvaluationPipeline, EvaluationResult
from meshforge.evaluation.preflight import PreflightChecker, PreflightResult
from meshforge.evaluation.metrics import MetricsCalculator, MeshQualityMetrics

__all__ = [
    "EvaluationPipeline",
    "EvaluationResult",
    "PreflightChecker",
    "PreflightResult",
    "MetricsCalculator",
    "MeshQualityMetrics",
]
