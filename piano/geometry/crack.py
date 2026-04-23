"""
Crack geometry definitions and mesh generation.

This module provides:
1. CrackGeometry base class - extensible for crack propagation
2. EdgeCrack / CenterCrack - specific crack configurations
3. CrackMeshGenerator - generates MFEM-compatible meshes with crack

Design Philosophy:
-----------------
The architecture separates geometry definition from mesh generation,
allowing future extension to crack propagation where:
- Crack tip position changes over time/load steps
- Mesh adapts to follow crack path
- History of crack path is tracked

For static analysis (current):
- Fixed crack geometry
- Refined mesh near crack tip to capture 1/sqrt(r) singularity

For future crack propagation:
- PropagatingCrack class will inherit from CrackGeometry
- Add methods: advance_tip(), get_crack_path(), compute_sif()
- Mesh regeneration or remeshing as crack grows
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class CrackType(Enum):
    """Types of crack configurations."""
    EDGE = "edge"           # Crack from edge into domain
    CENTER = "center"       # Internal crack (two tips)
    CORNER = "corner"       # Crack from corner
    BRANCHED = "branched"   # Branching crack (future)


class LoadingMode(Enum):
    """Fracture loading modes."""
    MODE_I = "mode_i"       # Opening mode (tensile)
    MODE_II = "mode_ii"     # Sliding mode (in-plane shear)
    MODE_III = "mode_iii"   # Tearing mode (out-of-plane shear)
    MIXED = "mixed"         # Combined modes


@dataclass
class CrackTip:
    """
    Represents a crack tip with position and direction.

    Designed to support crack propagation:
    - position: current tip location
    - direction: crack growth direction (tangent to crack)
    - history: list of previous positions (for propagation)
    """
    position: np.ndarray  # (x, y) coordinates
    direction: np.ndarray  # unit vector pointing in crack direction
    history: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.direction = np.asarray(self.direction, dtype=np.float64)
        # Normalize direction
        norm = np.linalg.norm(self.direction)
        if norm > 1e-10:
            self.direction = self.direction / norm

    def advance(self, distance: float, new_direction: Optional[np.ndarray] = None):
        """
        Advance crack tip (for future propagation).

        Args:
            distance: How far to advance
            new_direction: Optional new direction (for crack kinking)
        """
        self.history.append(self.position.copy())
        if new_direction is not None:
            self.direction = new_direction / np.linalg.norm(new_direction)
        self.position = self.position + distance * self.direction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "direction": self.direction.tolist(),
            "n_history": len(self.history),
        }


class CrackGeometry(ABC):
    """
    Abstract base class for crack geometry definitions.

    Designed for extensibility to crack propagation:
    - Subclasses define specific crack configurations
    - Tips can be updated for propagation
    - Provides interface for mesh generation

    Attributes:
        crack_type: Type of crack (edge, center, etc.)
        tips: List of crack tips
        width: Domain width
        height: Domain height
        loading_mode: Type of loading
    """

    def __init__(
        self,
        crack_type: CrackType,
        width: float = 1.0,
        height: float = 1.0,
        loading_mode: LoadingMode = LoadingMode.MODE_I,
    ):
        self.crack_type = crack_type
        self.tips: List[CrackTip] = []
        self.width = width
        self.height = height
        self.loading_mode = loading_mode

    @abstractmethod
    def get_crack_path(self) -> np.ndarray:
        """
        Get the crack path as array of points.

        Returns:
            Array of shape (N, 2) representing crack path
        """
        pass

    @abstractmethod
    def get_crack_length(self) -> float:
        """Get total crack length."""
        pass

    def get_tip_positions(self) -> np.ndarray:
        """Get all tip positions as array."""
        return np.array([tip.position for tip in self.tips])

    def is_point_on_crack(self, point: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if a point lies on the crack."""
        path = self.get_crack_path()
        for i in range(len(path) - 1):
            # Check distance to line segment
            p1, p2 = path[i], path[i + 1]
            d = self._point_to_segment_distance(point, p1, p2)
            if d < tolerance:
                return True
        return False

    @staticmethod
    def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance from point p to line segment ab."""
        ab = b - a
        ap = p - a
        t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-10), 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crack_type": self.crack_type.value,
            "tips": [t.to_dict() for t in self.tips],
            "width": self.width,
            "height": self.height,
            "loading_mode": self.loading_mode.value,
            "crack_length": self.get_crack_length(),
        }


class EdgeCrack(CrackGeometry):
    """
    Edge crack configuration: crack extends from one edge into domain.

    Common in fracture mechanics benchmarks (e.g., SENT specimen).

    Geometry:
        - Crack starts at left edge (x=0)
        - Extends horizontally into domain
        - Single crack tip

    Parameters:
        crack_length: Length of crack (a)
        crack_y: Y-position of crack (default: center)
        crack_angle: Angle of crack in degrees (0 = horizontal)
    """

    def __init__(
        self,
        crack_length: float = 0.3,
        crack_y: float = 0.5,
        crack_angle: float = 0.0,
        width: float = 1.0,
        height: float = 1.0,
        loading_mode: LoadingMode = LoadingMode.MODE_I,
    ):
        super().__init__(CrackType.EDGE, width, height, loading_mode)

        self.crack_length = crack_length
        self.crack_y = crack_y
        self.crack_angle = crack_angle

        # Compute crack tip position and direction
        angle_rad = np.radians(self.crack_angle)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        # Crack starts at left edge
        self._start = np.array([0.0, self.crack_y * self.height])
        tip_pos = self._start + self.crack_length * direction

        self.tips = [CrackTip(position=tip_pos, direction=direction)]

    def get_crack_path(self) -> np.ndarray:
        """Return crack path from edge to tip."""
        return np.array([self._start, self.tips[0].position])

    def get_crack_length(self) -> float:
        return self.crack_length


class CenterCrack(CrackGeometry):
    """
    Center crack configuration: internal crack with two tips.

    Common benchmark (e.g., center-cracked tension specimen).

    Geometry:
        - Crack centered in domain
        - Two tips, symmetric about center
        - Can be at an angle

    Parameters:
        crack_length: Total crack length (2a)
        center: Center of crack (default: domain center)
        crack_angle: Angle of crack in degrees
    """

    def __init__(
        self,
        crack_length: float = 0.4,
        center: Optional[Tuple[float, float]] = None,
        crack_angle: float = 0.0,
        width: float = 1.0,
        height: float = 1.0,
        loading_mode: LoadingMode = LoadingMode.MODE_I,
    ):
        super().__init__(CrackType.CENTER, width, height, loading_mode)

        self.crack_length = crack_length
        self.crack_angle = crack_angle

        if center is None:
            center = (self.width / 2, self.height / 2)
        self.center = center

        center_arr = np.array(self.center)
        angle_rad = np.radians(self.crack_angle)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        half_length = self.crack_length / 2

        # Two tips, opposite directions
        tip1_pos = center_arr + half_length * direction
        tip2_pos = center_arr - half_length * direction

        self.tips = [
            CrackTip(position=tip1_pos, direction=direction),
            CrackTip(position=tip2_pos, direction=-direction),
        ]

    def get_crack_path(self) -> np.ndarray:
        """Return crack path between two tips."""
        return np.array([self.tips[1].position, self.tips[0].position])

    def get_crack_length(self) -> float:
        return self.crack_length


class CrackMeshGenerator:
    """
    Generates MFEM-compatible meshes for crack problems.

    Features:
    - Refined mesh near crack tip (captures singularity)
    - Proper boundary tagging for BCs
    - Crack faces as internal boundaries

    Design for future crack propagation:
    - Mesh can be regenerated as crack grows
    - Or use mesh morphing to follow crack
    """

    def __init__(
        self,
        geometry: CrackGeometry,
        base_resolution: int = 20,
        tip_refinement: int = 3,
        tip_radius: float = 0.1,
    ):
        """
        Initialize mesh generator.

        Args:
            geometry: Crack geometry definition
            base_resolution: Base number of elements per unit length
            tip_refinement: Refinement levels near crack tip
            tip_radius: Radius of refined region around tip
        """
        self.geometry = geometry
        self.base_resolution = base_resolution
        self.tip_refinement = tip_refinement
        self.tip_radius = tip_radius

    def generate(self, output_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate mesh for crack geometry.

        Returns:
            vertices: (N, 2) array of vertex coordinates
            elements: (M, 3) array of triangle connectivity
            metadata: Dictionary with mesh info
        """
        from scipy.spatial import Delaunay

        W = self.geometry.width
        H = self.geometry.height

        # Generate base grid points
        nx = int(W * self.base_resolution)
        ny = int(H * self.base_resolution)

        points = []

        # Outer boundary points
        for i in range(nx + 1):
            points.append([i * W / nx, 0.0])  # bottom
            points.append([i * W / nx, H])    # top
        for j in range(1, ny):
            points.append([0.0, j * H / ny])  # left
            points.append([W, j * H / ny])    # right

        # Interior points (avoiding crack)
        crack_path = self.geometry.get_crack_path()

        for i in range(1, nx):
            for j in range(1, ny):
                p = np.array([i * W / nx, j * H / ny])
                # Skip points too close to crack
                if not self._near_crack(p, crack_path, tolerance=0.02):
                    points.append(p.tolist())

        # Refined points near crack tips
        for tip in self.geometry.tips:
            refined = self._generate_tip_refinement(tip.position)
            # Filter points outside domain or on crack
            for p in refined:
                if 0 < p[0] < W and 0 < p[1] < H:
                    if not self._near_crack(np.array(p), crack_path, tolerance=0.01):
                        points.append(list(p) if not isinstance(p, list) else p)

        # Add crack face points (both sides with small offset)
        crack_face_points = self._generate_crack_face_points(crack_path)
        points.extend(crack_face_points)

        vertices = np.array(points, dtype=np.float64)

        # Triangulate
        tri = Delaunay(vertices)

        # Filter triangles that cross the crack
        valid_triangles = []
        for simplex in tri.simplices:
            centroid = vertices[simplex].mean(axis=0)
            # Keep if centroid is not on crack
            if not self._near_crack(centroid, crack_path, tolerance=0.005):
                valid_triangles.append(simplex)

        elements = np.array(valid_triangles)

        metadata = {
            "n_vertices": len(vertices),
            "n_elements": len(elements),
            "crack_length": self.geometry.get_crack_length(),
            "crack_type": self.geometry.crack_type.value,
            "tip_positions": self.geometry.get_tip_positions().tolist(),
        }

        if output_path:
            self._write_mfem_mesh(vertices, elements, output_path)
            metadata["mesh_file"] = output_path

        return vertices, elements, metadata

    def _near_crack(self, point: np.ndarray, crack_path: np.ndarray,
                    tolerance: float) -> bool:
        """Check if point is near crack path."""
        point = np.asarray(point)
        for i in range(len(crack_path) - 1):
            d = CrackGeometry._point_to_segment_distance(
                point, crack_path[i], crack_path[i + 1]
            )
            if d < tolerance:
                return True
        return False

    def _generate_tip_refinement(self, tip_pos: np.ndarray) -> List[List[float]]:
        """Generate refined points around crack tip."""
        points = []

        for level in range(self.tip_refinement):
            r = self.tip_radius * (0.5 ** level)
            n_points = 8 * (level + 1)

            for i in range(n_points):
                theta = 2 * np.pi * i / n_points
                x = tip_pos[0] + r * np.cos(theta)
                y = tip_pos[1] + r * np.sin(theta)
                points.append([x, y])

        return points

    def _generate_crack_face_points(self, crack_path: np.ndarray) -> List[List[float]]:
        """Generate points along crack faces (both sides)."""
        points = []
        offset = 0.005  # Small offset from crack line

        for i in range(len(crack_path) - 1):
            p1, p2 = crack_path[i], crack_path[i + 1]

            # Normal to crack segment
            tangent = p2 - p1
            normal = np.array([-tangent[1], tangent[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-10)

            # Points along segment
            n_seg = max(3, int(np.linalg.norm(tangent) * self.base_resolution * 2))
            for j in range(1, n_seg):
                t = j / n_seg
                base = p1 + t * tangent
                # Both sides of crack
                points.append((base + offset * normal).tolist())
                points.append((base - offset * normal).tolist())

        return points

    def _write_mfem_mesh(self, vertices: np.ndarray, elements: np.ndarray,
                         output_path: str) -> None:
        """Write mesh in MFEM format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        W = self.geometry.width
        H = self.geometry.height
        eps = 1e-6

        with open(path, 'w') as f:
            f.write("MFEM mesh v1.0\n\n")
            f.write("dimension\n2\n\n")

            # Elements
            f.write(f"elements\n{len(elements)}\n")
            for tri in elements:
                f.write(f"1 2 {tri[0]} {tri[1]} {tri[2]}\n")
            f.write("\n")

            # Boundary elements (edges on domain boundary)
            boundary_edges = self._extract_boundary_edges(vertices, elements, W, H, eps)
            f.write(f"boundary\n{len(boundary_edges)}\n")
            for attr, v1, v2 in boundary_edges:
                f.write(f"{attr} 1 {v1} {v2}\n")
            f.write("\n")

            # Vertices
            f.write(f"vertices\n{len(vertices)}\n2\n")
            for v in vertices:
                f.write(f"{v[0]:.10f} {v[1]:.10f}\n")

    def _extract_boundary_edges(self, vertices: np.ndarray, elements: np.ndarray,
                                 W: float, H: float, eps: float) -> List[Tuple[int, int, int]]:
        """Extract boundary edges with attributes."""
        from collections import defaultdict

        # Count edge occurrences
        edge_count = defaultdict(int)
        edge_to_vertices = {}

        for tri in elements:
            for i in range(3):
                v1, v2 = tri[i], tri[(i + 1) % 3]
                edge = tuple(sorted([v1, v2]))
                edge_count[edge] += 1
                edge_to_vertices[edge] = (v1, v2)

        # Boundary edges appear once
        boundary_edges = []
        for edge, count in edge_count.items():
            if count == 1:
                v1, v2 = edge_to_vertices[edge]
                p1, p2 = vertices[v1], vertices[v2]
                mid = (p1 + p2) / 2

                # Assign boundary attribute
                # 1=bottom, 2=right, 3=top, 4=left, 5=crack
                if abs(mid[1]) < eps:
                    attr = 1  # bottom
                elif abs(mid[0] - W) < eps:
                    attr = 2  # right
                elif abs(mid[1] - H) < eps:
                    attr = 3  # top
                elif abs(mid[0]) < eps:
                    attr = 4  # left
                else:
                    attr = 5  # crack face (internal boundary)

                boundary_edges.append((attr, v1, v2))

        return boundary_edges


def generate_crack_mesh(
    crack_type: str = "edge",
    crack_length: float = 0.3,
    crack_angle: float = 0.0,
    width: float = 1.0,
    height: float = 1.0,
    resolution: int = 25,
    output_path: Optional[str] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convenience function to generate crack mesh.

    Args:
        crack_type: "edge" or "center"
        crack_length: Length of crack
        crack_angle: Angle in degrees
        width: Domain width
        height: Domain height
        resolution: Mesh resolution
        output_path: Optional path to save mesh
        **kwargs: Additional parameters for specific crack type

    Returns:
        vertices, elements, metadata
    """
    if crack_type == "edge":
        geometry = EdgeCrack(
            crack_length=crack_length,
            crack_y=kwargs.get("crack_y", 0.5),
            crack_angle=crack_angle,
            width=width,
            height=height,
        )
    elif crack_type == "center":
        geometry = CenterCrack(
            crack_length=crack_length,
            center=kwargs.get("center", None),
            crack_angle=crack_angle,
            width=width,
            height=height,
        )
    else:
        raise ValueError(f"Unknown crack type: {crack_type}")

    generator = CrackMeshGenerator(
        geometry=geometry,
        base_resolution=resolution,
        tip_refinement=kwargs.get("tip_refinement", 3),
        tip_radius=kwargs.get("tip_radius", 0.1),
    )

    return generator.generate(output_path)
