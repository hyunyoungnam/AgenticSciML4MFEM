"""
V-Notch geometry definitions and mesh generation.

This module provides:
1. VNotchGeometry - Sharp V-shaped notch configuration
2. VNotchMeshGenerator - Generates MFEM-compatible meshes with V-notch

V-notches are simpler than mathematical cracks:
- Finite opening angle (no true singularity, but high stress concentration)
- Easier to mesh consistently
- Common in experimental fracture specimens (Charpy, Izod)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VNotchGeometry:
    """
    V-shaped notch geometry configuration.

    The notch extends from the left edge into the plate with a specified
    depth and opening angle.

    Geometry:
        - Notch starts at left edge (x=0)
        - Opens symmetrically about the centerline
        - Tip is at (notch_depth, notch_y)

    Parameters:
        notch_depth: Depth of notch from edge (a)
        notch_angle: Total opening angle in degrees (e.g., 60 = 30° each side)
        notch_y: Y-position of notch centerline (default: 0.5 = center)
        width: Domain width
        height: Domain height
        tip_radius: Optional tip blunting radius (0 = sharp)
    """
    notch_depth: float = 0.3
    notch_angle: float = 60.0  # degrees
    notch_y: float = 0.5
    width: float = 1.0
    height: float = 1.0
    tip_radius: float = 0.0  # Sharp tip by default

    def __post_init__(self):
        # Validate parameters
        if self.notch_depth <= 0 or self.notch_depth >= self.width:
            raise ValueError(f"notch_depth must be in (0, {self.width})")
        if self.notch_angle <= 0 or self.notch_angle >= 180:
            raise ValueError("notch_angle must be in (0, 180) degrees")

    @property
    def tip_position(self) -> np.ndarray:
        """Get the notch tip position."""
        return np.array([self.notch_depth, self.notch_y * self.height])

    @property
    def half_angle_rad(self) -> float:
        """Half of the opening angle in radians."""
        return np.radians(self.notch_angle / 2)

    def get_notch_vertices(self) -> np.ndarray:
        """
        Get the vertices defining the V-notch shape.

        Returns:
            Array of shape (3, 2): [upper_edge, tip, lower_edge]
        """
        tip = self.tip_position
        half_angle = self.half_angle_rad

        # Notch edges at x=0
        # Upper edge: from tip, go back at +half_angle
        upper_y = tip[1] + self.notch_depth * np.tan(half_angle)
        lower_y = tip[1] - self.notch_depth * np.tan(half_angle)

        upper_edge = np.array([0.0, upper_y])
        lower_edge = np.array([0.0, lower_y])

        return np.array([upper_edge, tip, lower_edge])

    def get_notch_opening_at_edge(self) -> float:
        """Get the notch opening width at the left edge (x=0)."""
        return 2 * self.notch_depth * np.tan(self.half_angle_rad)

    def is_inside_notch(self, point: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Check if a point is inside the notch region.

        The notch region is the triangular area cut out from the plate.
        """
        x, y = point[0], point[1]
        tip = self.tip_position
        half_angle = self.half_angle_rad

        # Point must be to the left of tip
        if x > tip[0] + tolerance:
            return False

        # Distance from notch centerline
        dy = abs(y - tip[1])

        # At position x, the notch half-width is:
        # half_width = (tip[0] - x) * tan(half_angle)
        half_width = (tip[0] - x) * np.tan(half_angle)

        return dy < half_width - tolerance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notch_depth": self.notch_depth,
            "notch_angle": self.notch_angle,
            "notch_y": self.notch_y,
            "width": self.width,
            "height": self.height,
            "tip_radius": self.tip_radius,
            "tip_position": self.tip_position.tolist(),
            "opening_at_edge": self.get_notch_opening_at_edge(),
        }


class VNotchMeshGenerator:
    """
    Generates MFEM-compatible meshes for V-notch problems.

    Features:
    - Refined mesh near notch tip (captures stress concentration)
    - Proper boundary tagging for BCs
    - Notch faces as domain boundaries
    """

    # Boundary markers
    BOTTOM = 1
    RIGHT = 2
    TOP = 3
    LEFT_LOWER = 4  # Left edge below notch
    LEFT_UPPER = 5  # Left edge above notch
    NOTCH_LOWER = 6  # Lower notch face
    NOTCH_UPPER = 7  # Upper notch face

    def __init__(
        self,
        geometry: VNotchGeometry,
        base_resolution: int = 20,
        tip_refinement_levels: int = 4,
        tip_refinement_radius: float = 0.15,
    ):
        """
        Initialize mesh generator.

        Args:
            geometry: V-notch geometry definition
            base_resolution: Base number of elements per unit length
            tip_refinement_levels: Number of refinement levels near tip
            tip_refinement_radius: Radius of refined region around tip
        """
        self.geometry = geometry
        self.base_resolution = base_resolution
        self.tip_refinement_levels = tip_refinement_levels
        self.tip_refinement_radius = tip_refinement_radius

    def generate(
        self,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate mesh for V-notch geometry.

        Returns:
            vertices: (N, 2) array of vertex coordinates
            elements: (M, 3) array of triangle connectivity
            metadata: Dictionary with mesh info and boundary data
        """
        from scipy.spatial import Delaunay

        W = self.geometry.width
        H = self.geometry.height
        tip = self.geometry.tip_position
        notch_verts = self.geometry.get_notch_vertices()
        upper_edge_y = notch_verts[0, 1]
        lower_edge_y = notch_verts[2, 1]

        points = []
        boundary_markers = {}  # vertex_idx -> boundary_id

        # === Boundary points ===

        # Bottom edge (y=0)
        n_bottom = int(W * self.base_resolution)
        for i in range(n_bottom + 1):
            x = i * W / n_bottom
            idx = len(points)
            points.append([x, 0.0])
            boundary_markers[idx] = self.BOTTOM

        # Right edge (x=W)
        n_right = int(H * self.base_resolution)
        for j in range(1, n_right):
            y = j * H / n_right
            idx = len(points)
            points.append([W, y])
            boundary_markers[idx] = self.RIGHT

        # Top edge (y=H)
        for i in range(n_bottom, -1, -1):
            x = i * W / n_bottom
            idx = len(points)
            points.append([x, H])
            boundary_markers[idx] = self.TOP

        # Left edge - split by notch
        n_left = int(H * self.base_resolution)

        # Left edge below notch
        for j in range(n_left - 1, 0, -1):
            y = j * H / n_left
            if y < lower_edge_y - 0.01:
                idx = len(points)
                points.append([0.0, y])
                boundary_markers[idx] = self.LEFT_LOWER

        # Lower notch edge point
        idx = len(points)
        points.append([0.0, lower_edge_y])
        boundary_markers[idx] = self.NOTCH_LOWER

        # Lower notch face (from edge to tip)
        n_notch = max(5, int(self.geometry.notch_depth * self.base_resolution * 1.5))
        for i in range(1, n_notch):
            t = i / n_notch
            x = t * tip[0]
            y = lower_edge_y + t * (tip[1] - lower_edge_y)
            idx = len(points)
            points.append([x, y])
            boundary_markers[idx] = self.NOTCH_LOWER

        # Notch tip (add with slight offset to avoid degenerate triangles)
        idx = len(points)
        points.append([tip[0], tip[1]])
        boundary_markers[idx] = self.NOTCH_LOWER  # Tip belongs to notch

        # Upper notch face (from tip to edge)
        for i in range(1, n_notch):
            t = i / n_notch
            x = tip[0] - t * tip[0]
            y = tip[1] + t * (upper_edge_y - tip[1])
            idx = len(points)
            points.append([x, y])
            boundary_markers[idx] = self.NOTCH_UPPER

        # Upper notch edge point
        idx = len(points)
        points.append([0.0, upper_edge_y])
        boundary_markers[idx] = self.NOTCH_UPPER

        # Left edge above notch
        for j in range(1, n_left):
            y = j * H / n_left
            if y > upper_edge_y + 0.01:
                idx = len(points)
                points.append([0.0, y])
                boundary_markers[idx] = self.LEFT_UPPER

        # === Interior points ===
        n_interior_x = int(W * self.base_resolution * 0.8)
        n_interior_y = int(H * self.base_resolution * 0.8)

        for i in range(1, n_interior_x):
            for j in range(1, n_interior_y):
                x = i * W / n_interior_x
                y = j * H / n_interior_y
                p = np.array([x, y])

                # Skip points inside notch
                if self.geometry.is_inside_notch(p, tolerance=0.02):
                    continue

                # Skip points too close to tip (will add refined points)
                if np.linalg.norm(p - tip) < self.tip_refinement_radius * 0.5:
                    continue

                points.append([x, y])

        # === Refined points near tip ===
        for level in range(self.tip_refinement_levels):
            r = self.tip_refinement_radius * (0.5 ** level)
            n_ring = 12 * (level + 1)

            for i in range(n_ring):
                theta = 2 * np.pi * i / n_ring
                x = tip[0] + r * np.cos(theta)
                y = tip[1] + r * np.sin(theta)
                p = np.array([x, y])

                # Skip if outside domain or inside notch
                if x < 0 or x > W or y < 0 or y > H:
                    continue
                if self.geometry.is_inside_notch(p, tolerance=0.01):
                    continue

                points.append([x, y])

        # === Triangulate ===
        vertices = np.array(points, dtype=np.float64)
        tri = Delaunay(vertices)

        # Filter triangles inside notch
        valid_triangles = []
        for simplex in tri.simplices:
            centroid = vertices[simplex].mean(axis=0)
            if not self.geometry.is_inside_notch(centroid, tolerance=0.001):
                valid_triangles.append(simplex)

        elements = np.array(valid_triangles)

        metadata = {
            "n_vertices": len(vertices),
            "n_elements": len(elements),
            "notch_depth": self.geometry.notch_depth,
            "notch_angle": self.geometry.notch_angle,
            "tip_position": tip.tolist(),
            "boundary_markers": boundary_markers,
        }

        if output_path:
            self._write_mfem_mesh(vertices, elements, boundary_markers, output_path)
            metadata["mesh_file"] = output_path

        return vertices, elements, metadata

    def _write_mfem_mesh(
        self,
        vertices: np.ndarray,
        elements: np.ndarray,
        boundary_markers: Dict[int, int],
        output_path: str
    ) -> None:
        """Write mesh in MFEM format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Find boundary edges
        edge_count = {}
        for tri in elements:
            for i in range(3):
                v1, v2 = tri[i], tri[(i + 1) % 3]
                edge = tuple(sorted([v1, v2]))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        boundary_edges = []
        for edge, count in edge_count.items():
            if count == 1:  # Boundary edge
                v1, v2 = edge
                # Determine boundary marker from vertices
                m1 = boundary_markers.get(v1, 0)
                m2 = boundary_markers.get(v2, 0)
                marker = max(m1, m2) if m1 == m2 or min(m1, m2) == 0 else m1
                boundary_edges.append((marker, v1, v2))

        with open(output_path, 'w') as f:
            f.write("MFEM mesh v1.0\n\n")
            f.write("dimension\n2\n\n")

            f.write(f"elements\n{len(elements)}\n")
            for tri in elements:
                f.write(f"1 2 {tri[0]} {tri[1]} {tri[2]}\n")
            f.write("\n")

            f.write(f"boundary\n{len(boundary_edges)}\n")
            for marker, v1, v2 in boundary_edges:
                f.write(f"{marker} 1 {v1} {v2}\n")
            f.write("\n")

            f.write(f"vertices\n{len(vertices)}\n2\n")
            for v in vertices:
                f.write(f"{v[0]:.10f} {v[1]:.10f}\n")


def generate_vnotch_mesh(
    notch_depth: float = 0.3,
    notch_angle: float = 60.0,
    resolution: int = 20,
    output_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convenience function to generate V-notch mesh.

    Args:
        notch_depth: Depth of V-notch
        notch_angle: Opening angle in degrees
        resolution: Mesh resolution
        output_path: Optional path to save mesh

    Returns:
        vertices, elements, metadata
    """
    geometry = VNotchGeometry(
        notch_depth=notch_depth,
        notch_angle=notch_angle,
    )
    generator = VNotchMeshGenerator(
        geometry=geometry,
        base_resolution=resolution,
    )
    return generator.generate(output_path=output_path)
