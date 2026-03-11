"""
Solver interfaces module.

Provides abstract and concrete implementations for FEM solvers
using MFEM as the backend.
"""

from .base import (
    BoundaryCondition,
    BoundaryConditionType,
    MaterialProperties,
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
    "PhysicsConfig",
    "PhysicsType",
    "SolverInterface",
    "SolverResult",
    "MFEMSolver",
]
