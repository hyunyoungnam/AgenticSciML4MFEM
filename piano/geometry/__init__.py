"""
Geometry package for mesh generation.

Provides geometry definitions and mesh generators for various
problem domains, with focus on fracture mechanics.

Architecture designed for extensibility:
- StaticCrack: Fixed crack geometry (current)
- PropagatingCrack: Moving crack tip (future)
"""

from .crack import (
    CrackGeometry,
    EdgeCrack,
    CenterCrack,
    CrackMeshGenerator,
    generate_crack_mesh,
)

__all__ = [
    "CrackGeometry",
    "EdgeCrack",
    "CenterCrack",
    "CrackMeshGenerator",
    "generate_crack_mesh",
]
