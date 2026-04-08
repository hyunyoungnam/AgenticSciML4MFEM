"""
Mesh management module.

Provides abstract and concrete implementations for mesh management
across different FEM formats.
"""

from .base import MeshManager
from .mfem_manager import MFEMManager

__all__ = ["MeshManager", "MFEMManager"]
