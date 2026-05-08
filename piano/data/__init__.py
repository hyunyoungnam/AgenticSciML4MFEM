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
from .zero_copy import (
    mesh_vertices_to_numpy,
    gridfunction_nodal_to_numpy,
    scalar_gridfunction_to_numpy,
    as_float32,
    numpy_to_tensor,
    preallocate_float32,
)

__all__ = [
    "FEMSample",
    "FEMDataset",
    "DatasetConfig",
    "DatasetStatistics",
    "DatasetLoader",
    "MFEMDataLoader",
    # Zero-copy pipeline
    "mesh_vertices_to_numpy",
    "gridfunction_nodal_to_numpy",
    "scalar_gridfunction_to_numpy",
    "as_float32",
    "numpy_to_tensor",
    "preallocate_float32",
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
