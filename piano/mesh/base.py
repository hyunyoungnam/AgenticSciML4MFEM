"""
Abstract base class for mesh management.

Provides a unified interface for mesh operations across different FEM formats
(Abaqus .inp, MFEM .mesh, etc.).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np


class MeshManager(ABC):
    """
    Abstract base class for mesh management.

    Provides a unified interface for:
    - Loading mesh data (nodes, elements)
    - Updating node coordinates (for morphing)
    - Saving modified meshes

    Implementations should handle format-specific details internally.
    """

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the mesh manager.

        Args:
            file_path: Path to the mesh file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {file_path}")

    @abstractmethod
    def get_nodes(self) -> np.ndarray:
        """
        Get node coordinates.

        Returns:
            np.ndarray: Node coordinates with shape (N, dim) where dim is 2 or 3
        """
        pass

    @abstractmethod
    def get_node_ids(self) -> np.ndarray:
        """
        Get node IDs.

        Returns:
            np.ndarray: Node IDs with shape (N,)
        """
        pass

    @abstractmethod
    def get_elements(self) -> np.ndarray:
        """
        Get element connectivity.

        Returns:
            np.ndarray: Element connectivity with shape (M, nodes_per_element)
        """
        pass

    @abstractmethod
    def get_element_ids(self) -> np.ndarray:
        """
        Get element IDs.

        Returns:
            np.ndarray: Element IDs with shape (M,)
        """
        pass

    @abstractmethod
    def update_nodes(
        self,
        coords: np.ndarray,
        node_ids: Optional[np.ndarray] = None
    ) -> None:
        """
        Update node coordinates.

        Args:
            coords: New coordinates with shape (N, dim) or (K, dim) if node_ids provided
            node_ids: Optional array of node IDs to update. If None, updates all nodes.
        """
        pass

    @abstractmethod
    def save(self, output_path: Union[str, Path]) -> Path:
        """
        Save the mesh to a file.

        Args:
            output_path: Path to save the mesh

        Returns:
            Path: The path where the mesh was saved
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Get the spatial dimension of the mesh.

        Returns:
            int: 2 for 2D meshes, 3 for 3D meshes
        """
        pass

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """
        Get the number of nodes in the mesh.

        Returns:
            int: Number of nodes
        """
        pass

    @property
    @abstractmethod
    def num_elements(self) -> int:
        """
        Get the number of elements in the mesh.

        Returns:
            int: Number of elements
        """
        pass

    def get_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the bounding box of the mesh.

        Returns:
            tuple: (min_coords, max_coords) arrays of shape (dim,)
        """
        nodes = self.get_nodes()
        return nodes.min(axis=0), nodes.max(axis=0)

    def get_centroid(self) -> np.ndarray:
        """
        Get the centroid of the mesh.

        Returns:
            np.ndarray: Centroid coordinates of shape (dim,)
        """
        return self.get_nodes().mean(axis=0)
