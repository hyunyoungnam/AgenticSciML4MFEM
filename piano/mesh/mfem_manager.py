"""
MFEM mesh manager implementation.

Provides mesh management for MFEM .mesh files using the PyMFEM library.
"""

import ctypes
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .base import MeshManager

# Lazy import for optional dependency
_mfem = None


def _get_mfem():
    """Lazy import of mfem module."""
    global _mfem
    if _mfem is None:
        try:
            import mfem.ser as mfem
            _mfem = mfem
        except ImportError:
            raise ImportError(
                "PyMFEM is required for MFEM mesh support. "
                "Install with: pip install mfem"
            )
    return _mfem


class MFEMManager(MeshManager):
    """
    Mesh manager for MFEM .mesh files.

    Handles loading, modifying, and saving MFEM mesh files.
    Supports both 2D and 3D meshes.
    """

    # MFEM element type constants
    ELEMENT_TYPES = {
        0: "POINT",
        1: "SEGMENT",
        2: "TRIANGLE",
        3: "QUADRILATERAL",
        4: "TETRAHEDRON",
        5: "HEXAHEDRON",
        6: "WEDGE",
        7: "PYRAMID",
    }

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the MFEM mesh manager.

        Args:
            file_path: Path to the MFEM .mesh file
        """
        super().__init__(file_path)
        mfem = _get_mfem()

        # Load the mesh
        self._mesh = mfem.Mesh(str(self.file_path))
        self._dim = self._mesh.Dimension()
        self._space_dim = self._mesh.SpaceDimension()

        # Cache node and element data
        self._nodes: Optional[np.ndarray] = None
        self._node_ids: Optional[np.ndarray] = None
        self._elements: Optional[np.ndarray] = None
        self._element_ids: Optional[np.ndarray] = None
        self._modified = False

        # Extract data from mesh
        self._extract_mesh_data()

    def _extract_mesh_data(self) -> None:
        """Extract node and element data from the MFEM mesh object."""
        mfem = _get_mfem()

        # Extract vertices (nodes)
        num_vertices = self._mesh.GetNV()
        self._node_ids = np.arange(num_vertices, dtype=np.int32)

        # Get vertex coordinates
        # GetVertex returns a raw double* (SwigPyObject); use ctypes to read it
        self._nodes = np.zeros((num_vertices, self._space_dim), dtype=np.float64)
        for i in range(num_vertices):
            vertex = self._mesh.GetVertex(i)
            ptr = ctypes.cast(int(vertex), ctypes.POINTER(ctypes.c_double))
            for d in range(self._space_dim):
                self._nodes[i, d] = ptr[d]

        # Extract elements
        num_elements = self._mesh.GetNE()
        self._element_ids = np.arange(num_elements, dtype=np.int32)

        # Determine max nodes per element for connectivity array
        max_nodes = 0
        for i in range(num_elements):
            elem_vertices = self._mesh.GetElementVertices(i)
            max_nodes = max(max_nodes, len(elem_vertices))

        # Create connectivity array (pad with -1 for elements with fewer nodes)
        self._elements = np.full((num_elements, max_nodes), -1, dtype=np.int32)
        for i in range(num_elements):
            elem_vertices = self._mesh.GetElementVertices(i)
            for j, v in enumerate(elem_vertices):
                self._elements[i, j] = v

    def get_nodes(self) -> np.ndarray:
        """
        Get node coordinates.

        Returns:
            np.ndarray: Node coordinates with shape (N, dim)
        """
        return self._nodes.copy()

    def get_node_ids(self) -> np.ndarray:
        """
        Get node IDs.

        Returns:
            np.ndarray: Node IDs with shape (N,)
        """
        return self._node_ids.copy()

    def get_elements(self) -> np.ndarray:
        """
        Get element connectivity.

        Returns:
            np.ndarray: Element connectivity with shape (M, max_nodes_per_element)
                       Invalid entries are marked with -1
        """
        return self._elements.copy()

    def get_element_ids(self) -> np.ndarray:
        """
        Get element IDs.

        Returns:
            np.ndarray: Element IDs with shape (M,)
        """
        return self._element_ids.copy()

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
        if node_ids is None:
            # Update all nodes
            if coords.shape != self._nodes.shape:
                raise ValueError(
                    f"Coordinate shape {coords.shape} doesn't match "
                    f"expected shape {self._nodes.shape}"
                )
            self._nodes = coords.copy()
        else:
            # Update specific nodes
            if coords.shape[0] != len(node_ids):
                raise ValueError(
                    f"Number of coordinates {coords.shape[0]} doesn't match "
                    f"number of node IDs {len(node_ids)}"
                )
            for i, nid in enumerate(node_ids):
                if nid < 0 or nid >= len(self._nodes):
                    raise ValueError(f"Invalid node ID: {nid}")
                self._nodes[nid] = coords[i]

        # Update the MFEM mesh object
        self._update_mfem_mesh()
        self._modified = True

    def _update_mfem_mesh(self) -> None:
        """Update the MFEM mesh object with new node coordinates."""
        mfem = _get_mfem()

        # Get the nodes GridFunction from the mesh
        nodes_gf = self._mesh.GetNodes()

        if nodes_gf is not None:
            # High-order mesh: update the GridFunction
            fes = nodes_gf.FESpace()
            for i in range(self._mesh.GetNV()):
                for d in range(self._space_dim):
                    # For linear elements, vertex DOFs correspond directly
                    dof = fes.DofToVDof(i, d)
                    nodes_gf[dof] = self._nodes[i, d]
        else:
            # Linear mesh: update vertices directly via ctypes
            for i in range(self._mesh.GetNV()):
                vertex = self._mesh.GetVertex(i)
                ptr = ctypes.cast(int(vertex), ctypes.POINTER(ctypes.c_double))
                for d in range(self._space_dim):
                    ptr[d] = float(self._nodes[i, d])

    def save(self, output_path: Union[str, Path]) -> Path:
        """
        Save the mesh to a file.

        Args:
            output_path: Path to save the mesh

        Returns:
            Path: The path where the mesh was saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use MFEM's print function to save
        self._mesh.Print(str(output_path))

        return output_path

    @property
    def dimension(self) -> int:
        """
        Get the spatial dimension of the mesh.

        Returns:
            int: 2 for 2D meshes, 3 for 3D meshes
        """
        return self._space_dim

    @property
    def num_nodes(self) -> int:
        """
        Get the number of nodes in the mesh.

        Returns:
            int: Number of nodes
        """
        return len(self._nodes)

    @property
    def num_elements(self) -> int:
        """
        Get the number of elements in the mesh.

        Returns:
            int: Number of elements
        """
        return len(self._elements)

    @property
    def mesh(self):
        """
        Get the underlying MFEM mesh object.

        Returns:
            mfem.Mesh: The MFEM mesh object
        """
        return self._mesh

    @property
    def is_modified(self) -> bool:
        """
        Check if the mesh has been modified.

        Returns:
            bool: True if mesh has been modified
        """
        return self._modified

    def get_element_type(self, element_id: int) -> str:
        """
        Get the element type for a given element.

        Args:
            element_id: Element ID

        Returns:
            str: Element type name
        """
        if element_id < 0 or element_id >= self.num_elements:
            raise ValueError(f"Invalid element ID: {element_id}")

        elem_type = self._mesh.GetElementType(element_id)
        return self.ELEMENT_TYPES.get(elem_type, f"UNKNOWN({elem_type})")

    def get_boundary_attributes(self) -> np.ndarray:
        """
        Get boundary attributes (boundary IDs).

        Returns:
            np.ndarray: Array of boundary attributes
        """
        num_bdr = self._mesh.GetNBE()
        attrs = np.zeros(num_bdr, dtype=np.int32)
        for i in range(num_bdr):
            attrs[i] = self._mesh.GetBdrAttribute(i)
        return attrs

    def get_element_attributes(self) -> np.ndarray:
        """
        Get element attributes (material IDs).

        Returns:
            np.ndarray: Array of element attributes
        """
        attrs = np.zeros(self.num_elements, dtype=np.int32)
        for i in range(self.num_elements):
            attrs[i] = self._mesh.GetAttribute(i)
        return attrs
