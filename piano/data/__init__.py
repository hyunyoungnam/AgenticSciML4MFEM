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

# Conditionally import phase field generator (requires gmsh and dolfinx)
try:
    from .phase_field_generator import (
        PhaseFieldFEMConfig,
        ParameterBounds,
        generate_phase_field_sample,
        generate_phase_field_dataset,
        create_phase_field_dataset,
    )
    __all__.extend([
        "PhaseFieldFEMConfig",
        "ParameterBounds",
        "generate_phase_field_sample",
        "generate_phase_field_dataset",
        "create_phase_field_dataset",
    ])
except (ImportError, OSError):
    pass
