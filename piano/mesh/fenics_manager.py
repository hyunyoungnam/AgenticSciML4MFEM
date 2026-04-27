"""
FEniCS/DOLFINx mesh manager.

Provides mesh management interface for DOLFINx-based solvers,
supporting mesh loading from Gmsh and function space creation.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    from mpi4py import MPI
    import dolfinx
    from dolfinx import mesh as dfx_mesh
    from dolfinx import fem
    from dolfinx.io import gmshio
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

from .base import MeshManager


class FEniCSManager(MeshManager):
    """
    Mesh manager for DOLFINx (FEniCSx).

    Handles mesh loading from Gmsh files and provides access to
    DOLFINx mesh objects and function spaces.

    Note: This manager creates the mesh file path dynamically for
    cases where the mesh is generated in-memory by Gmsh.
    """

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        mesh: Optional["dolfinx.mesh.Mesh"] = None,
        cell_markers: Optional["dolfinx.mesh.MeshTags"] = None,
        facet_markers: Optional["dolfinx.mesh.MeshTags"] = None,
    ):
        """
        Initialize FEniCS mesh manager.

        Can be initialized with either a file path or a pre-existing mesh.

        Args:
            file_path: Path to mesh file (.msh or .xdmf)
            mesh: Pre-existing DOLFINx mesh
            cell_markers: Pre-existing cell markers
            facet_markers: Pre-existing facet markers
        """
        if not HAS_DOLFINX:
            raise ImportError(
                "DOLFINx is required for FEniCSManager. "
                "Install with: conda install -c conda-forge fenics-dolfinx"
            )

        self._mesh: Optional["dolfinx.mesh.Mesh"] = mesh
        self._cell_markers = cell_markers
        self._facet_markers = facet_markers
        self._function_spaces: dict = {}

        if file_path is not None:
            self.file_path = Path(file_path)
            if self.file_path.exists():
                self._load_mesh()
        elif mesh is not None:
            # Create a dummy path for parent class
            self.file_path = Path("in_memory_mesh")
        else:
            raise ValueError("Either file_path or mesh must be provided")

    def _load_mesh(self) -> None:
        """Load mesh from file."""
        suffix = self.file_path.suffix.lower()

        if suffix == ".msh":
            self._load_gmsh()
        elif suffix == ".xdmf":
            self._load_xdmf()
        else:
            raise ValueError(f"Unsupported mesh format: {suffix}")

    def _load_gmsh(self) -> None:
        """Load mesh from Gmsh .msh file."""
        self._mesh, self._cell_markers, self._facet_markers = gmshio.read_from_msh(
            str(self.file_path),
            MPI.COMM_WORLD,
            rank=0,
            gdim=2,
        )

    def _load_xdmf(self) -> None:
        """Load mesh from XDMF file."""
        from dolfinx.io import XDMFFile

        with XDMFFile(MPI.COMM_WORLD, str(self.file_path), "r") as xdmf:
            self._mesh = xdmf.read_mesh()
            try:
                self._cell_markers = xdmf.read_meshtags(self._mesh, "cell_markers")
            except RuntimeError:
                pass
            try:
                self._facet_markers = xdmf.read_meshtags(self._mesh, "facet_markers")
            except RuntimeError:
                pass

    @classmethod
    def from_gmsh_model(cls, model_name: str = "crack_mesh", gdim: int = 2) -> "FEniCSManager":
        """
        Create FEniCSManager from current Gmsh model in memory.

        Call this after gmsh.model.mesh.generate() but before gmsh.finalize().

        Args:
            model_name: Name of the Gmsh model
            gdim: Geometric dimension (2 or 3)

        Returns:
            FEniCSManager instance
        """
        import gmsh

        mesh, cell_markers, facet_markers = gmshio.model_to_mesh(
            gmsh.model,
            MPI.COMM_WORLD,
            rank=0,
            gdim=gdim,
        )

        return cls(mesh=mesh, cell_markers=cell_markers, facet_markers=facet_markers)

    def get_nodes(self) -> np.ndarray:
        """
        Get node coordinates.

        Returns:
            np.ndarray: Node coordinates with shape (N, dim)
        """
        return self._mesh.geometry.x[:, :self.dimension]

    def get_node_ids(self) -> np.ndarray:
        """
        Get node IDs.

        Returns:
            np.ndarray: Node IDs with shape (N,)
        """
        return np.arange(self._mesh.geometry.x.shape[0])

    def get_elements(self) -> np.ndarray:
        """
        Get element connectivity.

        Returns:
            np.ndarray: Element connectivity with shape (M, nodes_per_element)
        """
        topology = self._mesh.topology
        topology.create_connectivity(self.dimension, 0)
        cells = topology.connectivity(self.dimension, 0)

        # Convert to numpy array
        elements = []
        for i in range(cells.num_nodes):
            elements.append(cells.links(i))

        return np.array(elements, dtype=np.int64)

    def get_element_ids(self) -> np.ndarray:
        """
        Get element IDs.

        Returns:
            np.ndarray: Element IDs with shape (M,)
        """
        return np.arange(self._mesh.topology.index_map(self.dimension).size_local)

    def update_nodes(
        self,
        coords: np.ndarray,
        node_ids: Optional[np.ndarray] = None
    ) -> None:
        """
        Update node coordinates.

        Args:
            coords: New coordinates with shape (N, dim) or (K, dim) if node_ids provided
            node_ids: Optional array of node IDs to update
        """
        if node_ids is None:
            self._mesh.geometry.x[:, :self.dimension] = coords
        else:
            self._mesh.geometry.x[node_ids, :self.dimension] = coords

    def save(self, output_path: Union[str, Path]) -> Path:
        """
        Save the mesh to a file.

        Args:
            output_path: Path to save the mesh (.xdmf format)

        Returns:
            Path: The path where the mesh was saved
        """
        from dolfinx.io import XDMFFile

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with XDMFFile(MPI.COMM_WORLD, str(output_path), "w") as xdmf:
            xdmf.write_mesh(self._mesh)
            if self._cell_markers is not None:
                xdmf.write_meshtags(self._cell_markers, self._mesh.geometry)
            if self._facet_markers is not None:
                xdmf.write_meshtags(self._facet_markers, self._mesh.geometry)

        return output_path

    @property
    def dimension(self) -> int:
        """Get the spatial dimension of the mesh."""
        return self._mesh.topology.dim

    @property
    def num_nodes(self) -> int:
        """Get the number of nodes in the mesh."""
        return self._mesh.geometry.x.shape[0]

    @property
    def num_elements(self) -> int:
        """Get the number of elements in the mesh."""
        return self._mesh.topology.index_map(self.dimension).size_local

    @property
    def mesh(self) -> "dolfinx.mesh.Mesh":
        """Get the DOLFINx mesh object."""
        return self._mesh

    @property
    def cell_markers(self) -> Optional["dolfinx.mesh.MeshTags"]:
        """Get cell markers (physical groups)."""
        return self._cell_markers

    @property
    def facet_markers(self) -> Optional["dolfinx.mesh.MeshTags"]:
        """Get facet markers (boundary tags)."""
        return self._facet_markers

    def get_function_space(
        self,
        family: str = "Lagrange",
        degree: int = 1,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> "fem.FunctionSpace":
        """
        Get or create a function space on this mesh.

        Args:
            family: Finite element family (e.g., "Lagrange", "DG")
            degree: Polynomial degree
            shape: Shape for vector/tensor spaces (e.g., (2,) for 2D vector)

        Returns:
            DOLFINx FunctionSpace
        """
        key = (family, degree, shape)

        if key not in self._function_spaces:
            if shape is None:
                element = (family, degree)
            else:
                element = (family, degree, shape)

            self._function_spaces[key] = fem.functionspace(self._mesh, element)

        return self._function_spaces[key]

    def get_scalar_space(self, degree: int = 1) -> "fem.FunctionSpace":
        """Get scalar Lagrange function space."""
        return self.get_function_space("Lagrange", degree, None)

    def get_vector_space(self, degree: int = 1) -> "fem.FunctionSpace":
        """Get vector Lagrange function space."""
        return self.get_function_space("Lagrange", degree, (self.dimension,))

    def locate_boundary_dofs(
        self,
        V: "fem.FunctionSpace",
        marker: int,
    ) -> np.ndarray:
        """
        Locate DOFs on a boundary with given marker.

        Args:
            V: Function space
            marker: Boundary marker value

        Returns:
            Array of DOF indices on the boundary
        """
        if self._facet_markers is None:
            raise ValueError("No facet markers available")

        facets = self._facet_markers.find(marker)
        return fem.locate_dofs_topological(V, self.dimension - 1, facets)

    def create_boundary_measure(
        self,
        marker: int,
    ) -> "dolfinx.fem.form.Form":
        """
        Create integration measure for boundary with given marker.

        Args:
            marker: Boundary marker value

        Returns:
            DOLFINx measure (ds)
        """
        from dolfinx.fem import form
        import ufl

        if self._facet_markers is None:
            raise ValueError("No facet markers available")

        ds = ufl.Measure("ds", domain=self._mesh, subdomain_data=self._facet_markers)
        return ds(marker)
