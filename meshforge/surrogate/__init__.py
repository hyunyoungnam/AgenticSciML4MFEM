"""
Surrogate model module for adaptive learning.

Provides surrogate model interfaces and training utilities for predicting
FEM simulation outputs from input parameters.

Note: FNO/Transolver implementation is planned to replace DeepONet.
"""

from .base import SurrogateModel, SurrogateConfig, PredictionResult
from .trainer import SurrogateTrainer, TrainingConfig, TrainingResult
from .evaluator import SurrogateEvaluator, WeakRegion, UncertaintyAnalysis

__all__ = [
    "SurrogateModel",
    "SurrogateConfig",
    "PredictionResult",
    "SurrogateTrainer",
    "TrainingConfig",
    "TrainingResult",
    "SurrogateEvaluator",
    "WeakRegion",
    "UncertaintyAnalysis",
]
