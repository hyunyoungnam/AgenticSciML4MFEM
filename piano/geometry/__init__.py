"""
Geometry package for mesh generation.

Provides geometry definitions and mesh generators for various
problem domains, with focus on fracture mechanics.

Architecture designed for extensibility:
- StaticCrack: Fixed crack geometry (current)
- PropagatingCrack: Moving crack tip (future)
- VNotch: V-shaped notch for stress concentration studies
"""

from .crack import (
    CrackGeometry,
    EdgeCrack,
    CenterCrack,
    CrackMeshGenerator,
    generate_crack_mesh,
)

from .notch import (
    VNotchGeometry,
    VNotchMeshGenerator,
    generate_vnotch_mesh,
)

__all__ = [
    # Crack geometries
    "CrackGeometry",
    "EdgeCrack",
    "CenterCrack",
    "CrackMeshGenerator",
    "generate_crack_mesh",
    # V-notch geometries
    "VNotchGeometry",
    "VNotchMeshGenerator",
    "generate_vnotch_mesh",
]
