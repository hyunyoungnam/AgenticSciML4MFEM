"""
Mesh management module.

Provides abstract and concrete implementations for mesh management
across different FEM formats.
"""

from .base import MeshManager
from .mfem_manager import MFEMManager

__all__ = [
    "MeshManager",
    "MFEMManager",
]

# Conditionally import Gmsh components (OSError covers missing shared libs)
try:
    from .gmsh_generator import GmshMeshGenerator, GmshMeshConfig, generate_gmsh_crack_mesh
    __all__.extend(["GmshMeshGenerator", "GmshMeshConfig", "generate_gmsh_crack_mesh"])
except (ImportError, OSError):
    pass

# Conditionally import FEniCS components
try:
    from .fenics_manager import FEniCSManager
    __all__.append("FEniCSManager")
except (ImportError, OSError):
    pass
