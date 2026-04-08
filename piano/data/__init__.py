"""
Dataset management module.

Provides data structures and utilities for managing FEM simulation
datasets used for surrogate model training.
"""

from .dataset import (
    FEMSample,
    FEMDataset,
    DatasetConfig,
    DatasetStatistics,
)
from .loader import DatasetLoader, MFEMDataLoader

__all__ = [
    "FEMSample",
    "FEMDataset",
    "DatasetConfig",
    "DatasetStatistics",
    "DatasetLoader",
    "MFEMDataLoader",
]
