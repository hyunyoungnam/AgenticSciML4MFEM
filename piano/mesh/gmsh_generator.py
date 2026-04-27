"""
Gmsh mesh generator for phase field fracture problems.

Generates meshes with proper refinement near crack tips for accurate
phase field simulations using DOLFINx.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False

from ..geometry.crack import CrackGeometry, EdgeCrack, CenterCrack


@dataclass
class GmshMeshConfig:
    """
    Configuration for Gmsh mesh generation.

    Attributes:
        base_size: Base element size (fraction of domain)
        tip_size: Element size at crack tip (should be l_0 / 5 for phase field)
        tip_radius: Radius of refined region around tip
        growth_rate: Element size growth rate from tip
        mesh_order: Element order (1 = linear, 2 = quadratic)
        algorithm: Meshing algorithm (default: Delaunay)
    """
    base_size: float = 0.02      # ~1/50 of unit domain
    tip_size: float = 0.003      # l_0 / 5 for l_0 = 0.015
    tip_radius: float = 0.1      # Refined region radius
    growth_rate: float = 1.1     # Size field growth rate
    mesh_order: int = 1          # Linear elements
    algorithm: int = 5           # Delaunay (5) or Frontal-Delaunay (6)


class GmshMeshGenerator:
    """
    Generates Gmsh meshes for fracture problems with crack tip refinement.

    Uses Gmsh size fields for smooth mesh grading from fine elements
    at crack tips to coarse elements in the bulk.
    """

    def __init__(
        self,
        geometry: CrackGeometry,
        config: Optional[GmshMeshConfig] = None,
    ):
        """
        Initialize mesh generator.

        Args:
            geometry: Crack geometry definition
            config: Mesh generation configuration
        """
        if not HAS_GMSH:
            raise ImportError(
                "gmsh is required for GmshMeshGenerator. "
                "Install with: conda install -c conda-forge gmsh"
            )

        self.geometry = geometry
        self.config = config or GmshMeshConfig()
        self._mesh_generated = False

    def generate(
        self,
        output_path: Optional[Union[str, Path]] = None,
        show_gui: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate mesh for the crack geometry.

        Args:
            output_path: Path to save mesh file (.msh format)
            show_gui: Whether to show Gmsh GUI (for debugging)

        Returns:
            vertices: (N, 2) array of vertex coordinates
            elements: (M, 3) array of triangle connectivity
            metadata: Dictionary with mesh info
        """
        gmsh.initialize()
        gmsh.clear()
        gmsh.model.add("crack_mesh")

        try:
            # Create geometry
            self._create_domain()
            self._create_crack()

            # Set up mesh size fields
            self._setup_size_fields()

            # Generate mesh
            gmsh.option.setNumber("Mesh.Algorithm", self.config.algorithm)
            gmsh.option.setNumber("Mesh.ElementOrder", self.config.mesh_order)
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)

            if show_gui:
                gmsh.fltk.run()

            # Extract mesh data
            vertices, elements = self._extract_mesh_data()

            metadata = {
                "n_vertices": len(vertices),
                "n_elements": len(elements),
                "crack_length": self.geometry.get_crack_length(),
                "crack_type": self.geometry.crack_type.value,
                "tip_positions": self.geometry.get_tip_positions().tolist(),
                "base_size": self.config.base_size,
                "tip_size": self.config.tip_size,
            }

            # Save if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                gmsh.write(str(output_path))
                metadata["mesh_file"] = str(output_path)

            self._mesh_generated = True
            return vertices, elements, metadata

        finally:
            gmsh.finalize()

    def _create_domain(self) -> None:
        """Create the rectangular domain."""
        W = self.geometry.width
        H = self.geometry.height

        # Domain corners
        p1 = gmsh.model.geo.addPoint(0, 0, 0, self.config.base_size)
        p2 = gmsh.model.geo.addPoint(W, 0, 0, self.config.base_size)
        p3 = gmsh.model.geo.addPoint(W, H, 0, self.config.base_size)
        p4 = gmsh.model.geo.addPoint(0, H, 0, self.config.base_size)

        # Domain boundary lines
        l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
        l2 = gmsh.model.geo.addLine(p2, p3)  # right
        l3 = gmsh.model.geo.addLine(p3, p4)  # top
        l4 = gmsh.model.geo.addLine(p4, p1)  # left

        # Store for boundary conditions
        self._boundary_lines = {
            "bottom": l1,
            "right": l2,
            "top": l3,
            "left": l4,
        }

        # Domain curve loop and surface
        loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        self._domain_surface = gmsh.model.geo.addPlaneSurface([loop])

        # Physical groups for boundary conditions
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [l1], 1, "bottom")
        gmsh.model.addPhysicalGroup(1, [l2], 2, "right")
        gmsh.model.addPhysicalGroup(1, [l3], 3, "top")
        gmsh.model.addPhysicalGroup(1, [l4], 4, "left")
        gmsh.model.addPhysicalGroup(2, [self._domain_surface], 1, "domain")

    def _create_crack(self) -> None:
        """Create crack as embedded line in domain."""
        crack_path = self.geometry.get_crack_path()

        # Add crack points
        crack_points = []
        for i, point in enumerate(crack_path):
            # Use fine mesh size at crack tip, coarser at other points
            is_tip = self._is_crack_tip(point)
            size = self.config.tip_size if is_tip else self.config.base_size
            p = gmsh.model.geo.addPoint(point[0], point[1], 0, size)
            crack_points.append(p)

        # Add crack line segments
        crack_lines = []
        for i in range(len(crack_points) - 1):
            line = gmsh.model.geo.addLine(crack_points[i], crack_points[i + 1])
            crack_lines.append(line)

        # Embed crack in surface (creates internal boundary)
        gmsh.model.geo.synchronize()
        for line in crack_lines:
            gmsh.model.mesh.embed(1, [line], 2, self._domain_surface)

        # Physical group for crack
        gmsh.model.addPhysicalGroup(1, crack_lines, 5, "crack")

        self._crack_lines = crack_lines
        self._crack_points = crack_points

    def _is_crack_tip(self, point: np.ndarray) -> bool:
        """Check if point is a crack tip."""
        point = np.asarray(point)
        for tip in self.geometry.tips:
            if np.linalg.norm(point - tip.position) < 1e-10:
                return True
        return False

    def _setup_size_fields(self) -> None:
        """Set up Gmsh size fields for mesh refinement."""
        fields = []

        # Distance field from crack tips
        for i, tip in enumerate(self.geometry.tips):
            # Point for distance calculation
            tip_point = gmsh.model.geo.addPoint(
                tip.position[0], tip.position[1], 0, self.config.tip_size
            )
            gmsh.model.geo.synchronize()

            # Distance field
            dist_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_field, "PointsList", [tip_point])

            # Threshold field for graded refinement
            thresh_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", self.config.tip_size)
            gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", self.config.base_size)
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", self.config.tip_radius)

            fields.append(thresh_field)

        # Distance field from crack line for general refinement
        if self._crack_lines:
            crack_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(crack_dist, "CurvesList", self._crack_lines)

            crack_thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(crack_thresh, "InField", crack_dist)
            gmsh.model.mesh.field.setNumber(crack_thresh, "SizeMin", self.config.tip_size * 2)
            gmsh.model.mesh.field.setNumber(crack_thresh, "SizeMax", self.config.base_size)
            gmsh.model.mesh.field.setNumber(crack_thresh, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(crack_thresh, "DistMax", self.config.tip_radius / 2)

            fields.append(crack_thresh)

        # Combine all fields using Min
        if fields:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    def _extract_mesh_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract vertices and elements from Gmsh."""
        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        # Reshape coordinates to (N, 3) then take (N, 2) for 2D
        node_coords = node_coords.reshape(-1, 3)[:, :2]

        # Create node tag to index mapping
        tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}

        # Get triangular elements (type 2 in Gmsh)
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)

        triangles = []
        for elem_type, elem_nodes in zip(elem_types, elem_node_tags):
            if elem_type == 2:  # 3-node triangle
                elem_nodes = elem_nodes.reshape(-1, 3)
                for tri in elem_nodes:
                    triangles.append([tag_to_idx[t] for t in tri])

        vertices = np.array(node_coords, dtype=np.float64)
        elements = np.array(triangles, dtype=np.int64)

        return vertices, elements

    def generate_for_phase_field(
        self,
        l_0: float,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate mesh optimized for phase field with regularization length l_0.

        Automatically sets tip_size = l_0 / 5 as recommended for phase field.

        Args:
            l_0: Regularization length scale
            output_path: Path to save mesh file

        Returns:
            vertices, elements, metadata
        """
        # Update config for phase field requirements
        self.config.tip_size = l_0 / 5.0
        self.config.tip_radius = 3.0 * l_0  # Refined region should be ~3*l_0

        return self.generate(output_path)


def generate_gmsh_crack_mesh(
    crack_type: str = "edge",
    crack_length: float = 0.3,
    width: float = 1.0,
    height: float = 1.0,
    l_0: float = 0.015,
    output_path: Optional[str] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Convenience function to generate Gmsh mesh for crack problems.

    Args:
        crack_type: "edge" or "center"
        crack_length: Length of crack
        width: Domain width
        height: Domain height
        l_0: Phase field regularization length (sets element size)
        output_path: Path to save mesh
        **kwargs: Additional arguments for crack geometry

    Returns:
        vertices, elements, metadata
    """
    if crack_type == "edge":
        geometry = EdgeCrack(
            crack_length=crack_length,
            crack_y=kwargs.get("crack_y", 0.5),
            width=width,
            height=height,
        )
    elif crack_type == "center":
        geometry = CenterCrack(
            crack_length=crack_length,
            center=kwargs.get("center", None),
            width=width,
            height=height,
        )
    else:
        raise ValueError(f"Unknown crack type: {crack_type}")

    config = GmshMeshConfig(
        base_size=kwargs.get("base_size", 1.0 / 50),
        tip_size=l_0 / 5,
        tip_radius=3 * l_0,
    )

    generator = GmshMeshGenerator(geometry, config)
    return generator.generate(output_path)
