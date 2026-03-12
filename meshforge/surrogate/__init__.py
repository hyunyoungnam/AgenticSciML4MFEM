"""
Surrogate model module for adaptive learning.

Provides DeepONet-based surrogate models using DeepXDE for predicting
FEM simulation outputs from input parameters.
"""

from .base import SurrogateModel, SurrogateConfig, PredictionResult
from .deeponet import DeepONetSurrogate
from .trainer import SurrogateTrainer, TrainingConfig, TrainingResult
from .evaluator import SurrogateEvaluator, WeakRegion, UncertaintyAnalysis

__all__ = [
    "SurrogateModel",
    "SurrogateConfig",
    "PredictionResult",
    "DeepONetSurrogate",
    "SurrogateTrainer",
    "TrainingConfig",
    "TrainingResult",
    "SurrogateEvaluator",
    "WeakRegion",
    "UncertaintyAnalysis",
]
