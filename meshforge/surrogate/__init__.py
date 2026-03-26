"""
Surrogate model module for adaptive learning.

Provides surrogate model interfaces and training utilities for predicting
FEM simulation outputs from input parameters using Transolver neural operator.
"""

from .base import (
    SurrogateModel,
    SurrogateConfig,
    TransolverConfig,
    EnsembleConfig,
    PredictionResult,
    SurrogateType,
)
from .transolver import TransolverModel, PhysicsAttention
from .ensemble import EnsembleModel
from .trainer import SurrogateTrainer, TrainingConfig, TrainingResult
from .evaluator import SurrogateEvaluator, WeakRegion, UncertaintyAnalysis

__all__ = [
    # Base
    "SurrogateModel",
    "SurrogateConfig",
    "TransolverConfig",
    "EnsembleConfig",
    "PredictionResult",
    "SurrogateType",
    # Models
    "TransolverModel",
    "PhysicsAttention",
    "EnsembleModel",
    # Training
    "SurrogateTrainer",
    "TrainingConfig",
    "TrainingResult",
    # Evaluation
    "SurrogateEvaluator",
    "WeakRegion",
    "UncertaintyAnalysis",
]
