"""
Solver interfaces module.

Provides abstract and concrete implementations for FEM solvers
using MFEM as the backend.
"""

from .base import (
    BoundaryCondition,
    BoundaryConditionType,
    MaterialProperties,
    PhaseFieldConfig,
    PhysicsConfig,
    PhysicsType,
    SolverInterface,
    SolverResult,
)
from .mfem_solver import MFEMSolver

__all__ = [
    "BoundaryCondition",
    "BoundaryConditionType",
    "MaterialProperties",
    "PhaseFieldConfig",
    "PhysicsConfig",
    "PhysicsType",
    "SolverInterface",
    "SolverResult",
    "MFEMSolver",
]

# Conditionally import FEniCS components
try:
    from .fenics_phase_field import FEniCSPhaseFieldSolver, PhaseFieldState
    __all__.extend(["FEniCSPhaseFieldSolver", "PhaseFieldState"])
except ImportError:
    pass
