"""
R-adaptivity using TMOP for error-driven mesh adaptation.

Implements r-adaptivity (node relocation) using MFEM's Target-Matrix Optimization
Paradigm (TMOP). Nodes are redistributed to cluster in high-error regions based
on the surrogate model's error map.

Key concepts:
1. Target matrix is constructed as a spatial function of error - smaller target
   element size in high-error regions attracts nodes there.
2. TMOP's barrier functions prevent element tangling and ensure mesh validity.
3. The error field from surrogate model (e.g., FNO/Transolver) drives the adaptation.

Reference:
    Dobrev et al., "The Target-Matrix Optimization Paradigm for High-Order Meshes"
    https://mfem.org/mesh-optimization/
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..mesh.base import MeshManager


@dataclass
class AdaptivityConfig:
    """
    Configuration for r-adaptivity.

    Attributes:
        size_scale_min: Minimum scaling factor for target element size (in high-error regions)
        size_scale_max: Maximum scaling factor for target element size (in low-error regions)
        error_threshold: Error values below this are considered low-error
        smoothing_iterations: Number of Laplacian smoothing iterations for error field
        target_type: TMOP target type ("ideal_shape_equal_size", "given_shape_and_size")
        quality_metric: TMOP quality metric ("shape", "size", "shape_and_size")
        max_iterations: Maximum Newton iterations
        tolerance: Newton solver tolerance
        verbosity: Output verbosity level
    """
    # Size scaling based on error
    size_scale_min: float = 0.5  # Target size in high-error regions (attracts nodes)
    size_scale_max: float = 1.5  # Target size in low-error regions (repels nodes)
    error_threshold: float = 0.1  # Normalize errors relative to max

    # Error field processing
    smoothing_iterations: int = 2  # Smooth error field to avoid oscillations

    # TMOP settings
    target_type: str = "ideal_shape_given_size"
    quality_metric: str = "shape_and_size"
    max_iterations: int = 200
    tolerance: float = 1e-8
    verbosity: int = 0

    # Boundary handling
    fix_boundary: bool = True  # Fix boundary nodes during adaptation

    # Limiting: quadratic penalty ||x - x_init||^2 * coeff added to energy.
    # Larger values bound node movement more tightly, preventing inversions.
    limiting_coeff: float = 100.0


@dataclass
class AdaptivityResult:
    """
    Result of r-adaptivity.

    Attributes:
        success: Whether adaptation succeeded
        coords_adapted: New node coordinates after adaptation
        quality_before: Mesh quality metrics before adaptation
        quality_after: Mesh quality metrics after adaptation
        iterations: Number of iterations used
        error_message: Error message if failed
    """
    success: bool
    coords_adapted: np.ndarray
    quality_before: Dict[str, float] = field(default_factory=dict)
    quality_after: Dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    error_message: Optional[str] = None


class TMOPAdaptivity:
    """
    R-adaptivity using TMOP for error-driven mesh adaptation.

    Uses the surrogate model's error map to construct spatially-varying
    target element sizes, driving nodes to cluster in high-error regions
    while maintaining mesh validity through barrier functions.

    Example:
        # Get error field from surrogate model
        error_field = surrogate.compute_pointwise_error(coords, params)

        # Create adaptivity engine
        adaptivity = TMOPAdaptivity(config)

        # Adapt mesh based on error
        result = adaptivity.adapt(mesh_manager, error_field)

        if result.success:
            print(f"Adapted mesh, quality improved from "
                  f"{result.quality_before['min_quality']:.3f} to "
                  f"{result.quality_after['min_quality']:.3f}")
    """

    def __init__(self, config: Optional[AdaptivityConfig] = None):
        """
        Initialize TMOP adaptivity engine.

        Args:
            config: Adaptivity configuration. Uses defaults if None.
        """
        self.config = config or AdaptivityConfig()

    def adapt(
        self,
        manager: MeshManager,
        error_field: np.ndarray,
        fixed_nodes: Optional[np.ndarray] = None,
    ) -> AdaptivityResult:
        """
        Adapt mesh based on error field using TMOP r-adaptivity.

        Nodes are relocated to cluster in high-error regions. The target
        element size is inversely related to local error - high error
        regions get smaller target sizes, attracting nodes.

        Args:
            manager: MeshManager with loaded mesh
            error_field: (N,) array of error values at each node.
                        Higher values indicate regions needing more resolution.
            fixed_nodes: Optional (M,) array of node indices to keep fixed.
                        If None and config.fix_boundary=True, boundary nodes are fixed.

        Returns:
            AdaptivityResult with adapted coordinates and quality metrics
        """
        try:
            import mfem.ser as mfem
        except ImportError:
            return AdaptivityResult(
                success=False,
                coords_adapted=np.array([]),
                error_message="PyMFEM not installed. Install with: pip install mfem"
            )

        try:
            coords = manager.get_nodes()
            if coords is None or len(coords) == 0:
                return AdaptivityResult(
                    success=False,
                    coords_adapted=np.array([]),
                    error_message="Mesh has no nodes"
                )

            n_nodes = len(coords)

            if len(error_field) != n_nodes:
                return AdaptivityResult(
                    success=False,
                    coords_adapted=coords.copy(),
                    error_message=f"Error field length ({len(error_field)}) != "
                                 f"number of nodes ({n_nodes})"
                )

            from ..mesh.mfem_manager import MFEMManager
            if not isinstance(manager, MFEMManager):
                return AdaptivityResult(
                    success=False,
                    coords_adapted=coords.copy(),
                    error_message="TMOPAdaptivity requires MFEMManager"
                )

            mesh = manager.mesh
            quality_before = self._compute_quality(mesh)

            processed_error = self._process_error_field(error_field, coords, mesh)
            target_sizes = self._error_to_target_size(processed_error)

            if fixed_nodes is None and self.config.fix_boundary:
                fixed_nodes = self._get_boundary_nodes(mesh)

            coords_adapted, iterations = self._run_tmop(
                mesh, target_sizes, fixed_nodes
            )

            manager.update_nodes(coords_adapted)
            quality_after = self._compute_quality(mesh)

            return AdaptivityResult(
                success=True,
                coords_adapted=coords_adapted,
                quality_before=quality_before,
                quality_after=quality_after,
                iterations=iterations,
            )

        except Exception as e:
            return AdaptivityResult(
                success=False,
                coords_adapted=manager.get_nodes() if manager else np.array([]),
                error_message=f"TMOP adaptivity error: {str(e)}"
            )

    def _process_error_field(
        self,
        error_field: np.ndarray,
        coords: np.ndarray,
        mesh
    ) -> np.ndarray:
        """Process error field: normalize and smooth."""
        error = np.maximum(error_field, 0.0)

        max_error = np.max(error)
        if max_error > 1e-12:
            error = error / max_error
        else:
            error = np.ones_like(error) * 0.5

        for _ in range(self.config.smoothing_iterations):
            error = self._laplacian_smooth(error, coords, mesh)

        return error

    def _laplacian_smooth(
        self,
        field: np.ndarray,
        coords: np.ndarray,
        mesh
    ) -> np.ndarray:
        """Apply one iteration of Laplacian smoothing."""
        n_nodes = len(field)
        smoothed = field.copy()

        neighbor_sum = np.zeros(n_nodes)
        neighbor_count = np.zeros(n_nodes)

        n_elements = mesh.GetNE()
        for e in range(n_elements):
            vertices = mesh.GetElementVertices(e)
            for v in vertices:
                for u in vertices:
                    if u != v:
                        neighbor_sum[v] += field[u]
                        neighbor_count[v] += 1

        mask = neighbor_count > 0
        smoothed[mask] = 0.5 * field[mask] + 0.5 * neighbor_sum[mask] / neighbor_count[mask]

        return smoothed

    def _error_to_target_size(self, error: np.ndarray) -> np.ndarray:
        """
        Convert normalized error field to target element size field.

        High error → small target size (attracts nodes)
        Low error → large target size (repels nodes)
        """
        size_min = self.config.size_scale_min
        size_max = self.config.size_scale_max
        return size_max - (size_max - size_min) * error

    def _get_boundary_nodes(self, mesh) -> np.ndarray:
        """
        Get indices of boundary nodes using topological edge detection.

        An edge shared by exactly one element is a boundary edge; all its
        vertices are boundary nodes.  This correctly identifies ALL boundary
        nodes even when the mesh file only annotates a subset of boundary
        elements (as is common with PyMFEM-generated plate-with-hole meshes).
        """
        from collections import defaultdict

        # Count how many elements own each edge (sorted tuple of vertex indices)
        edge_count: dict = defaultdict(int)
        n_elements = mesh.GetNE()
        for e in range(n_elements):
            verts = list(mesh.GetElementVertices(e))
            n = len(verts)
            for i in range(n):
                edge = tuple(sorted((verts[i], verts[(i + 1) % n])))
                edge_count[edge] += 1

        # Boundary edges are shared by exactly one element
        boundary_nodes: set = set()
        for edge, count in edge_count.items():
            if count == 1:
                boundary_nodes.update(edge)

        return np.array(sorted(boundary_nodes), dtype=int)

    def _run_tmop(
        self,
        mesh,
        target_sizes: np.ndarray,
        fixed_nodes: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """
        Run TMOP optimization using DiscreteAdaptTC with nodal size field.

        Args:
            mesh: MFEM mesh (must have high-order nodes via SetCurvature)
            target_sizes: Target size scaling per node
            fixed_nodes: Indices of nodes to keep fixed

        Returns:
            Tuple of (adapted coordinates, number of iterations)
        """
        import mfem.ser as mfem
        tmop = mfem.tmop

        dim = mesh.Dimension()

        # Ensure mesh has high-order nodes (required for TMOP)
        if mesh.GetNodes() is None:
            mesh.SetCurvature(1)

        # Use the mesh's own nodes GridFunction as the optimization variable.
        # This avoids byNODES/byVDIM ordering confusion: mesh.GetNodes(copy)
        # raw-copies the vector ignoring FESpace ordering differences.
        # Operating on the mesh's own GF also means no SetNodes call is needed.
        x = mesh.GetNodes()
        fes_mesh = x.FESpace()

        # Build H1 FE space for scalar nodal size field
        fec_size = mfem.H1_FECollection(1, dim)
        fes_size = mfem.FiniteElementSpace(mesh, fec_size)

        # Scale target_sizes to physical Jacobian-determinant units.
        # DiscreteAdaptTC.SetSerialDiscreteTargetSize expects det(J) values
        # (≈ 2 × element area for triangles), not dimensionless scale factors.
        mean_det = self._compute_mean_det(mesh)

        size_gf = mfem.GridFunction(fes_size)
        n = min(len(target_sizes), size_gf.Size())
        for i in range(n):
            size_gf[i] = float(target_sizes[i]) * mean_det
        for i in range(n, size_gf.Size()):
            size_gf[i] = mean_det

        # x_init: fixed copy of initial node positions for limiting reference
        # and as the fixed reference for DiscreteAdaptTC.
        x_init = mfem.GridFunction(fes_mesh)
        for i in range(x.Size()):
            x_init[i] = x[i]

        # Setup DiscreteAdaptTC with fixed reference x_init
        datc = tmop.DiscreteAdaptTC(tmop.TargetConstructor.IDEAL_SHAPE_GIVEN_SIZE)
        adv = tmop.AdvectorCG()
        datc.SetAdaptivityEvaluator(adv)
        datc.SetSerialDiscreteTargetSize(size_gf)
        datc.SetNodes(x_init)

        # Choose TMOP quality metric
        if dim == 2:
            if self.config.quality_metric == "shape":
                mu = tmop.TMOP_Metric_002()
            elif self.config.quality_metric == "size":
                mu = tmop.TMOP_Metric_080()
            else:
                mu = tmop.TMOP_Metric_077()
        else:
            if self.config.quality_metric == "shape":
                mu = tmop.TMOP_Metric_302()
            elif self.config.quality_metric == "size":
                mu = tmop.TMOP_Metric_316()
            else:
                mu = tmop.TMOP_Metric_321()

        tmop_integ = tmop.TMOP_Integrator(mu, datc)

        # Limiting: quadratic penalty ||x - x_init||^2 * coeff bounds node
        # movement and keeps the TMOP Hessian SPD (prevents inversions).
        limiting_coeff = mfem.ConstantCoefficient(self.config.limiting_coeff)
        tmop_integ.EnableLimiting(x_init, limiting_coeff)

        # Nonlinear form on the mesh's own FESpace
        a = mfem.NonlinearForm(fes_mesh)
        a.AddDomainIntegrator(tmop_integ)

        # Fix essential DOFs using DofToVDof — ordering-agnostic.
        if fixed_nodes is not None and len(fixed_nodes) > 0:
            ess_tdof_list = mfem.intArray()
            for node_idx in fixed_nodes:
                for d in range(dim):
                    dof = fes_mesh.DofToVDof(int(node_idx), d)
                    if 0 <= dof < fes_mesh.GetTrueVSize():
                        ess_tdof_list.Append(dof)
            a.SetEssentialTrueDofs(ess_tdof_list)

        # CG inner solver: TMOP Hessian + limiting is SPD on a valid mesh.
        lin_solver = mfem.CGSolver()
        lin_solver.SetRelTol(1e-12)
        lin_solver.SetAbsTol(0.0)
        lin_solver.SetMaxIter(500)
        lin_solver.SetPrintLevel(-1)

        # TMOPNewtonSolver has barrier-aware backtracking line search that
        # prevents Newton steps from crossing det(J)=0 (inverting elements).
        # Build integration rule matching the mesh's element geometry.
        geom = mesh.GetElementGeometry(0)
        ir = mfem.IntRules.Get(geom, 8)
        solver = tmop.TMOPNewtonSolver(ir)
        solver.SetMaxIter(self.config.max_iterations)
        solver.SetRelTol(self.config.tolerance)
        solver.SetAbsTol(0.0)
        solver.SetPrintLevel(self.config.verbosity)
        solver.SetOperator(a)
        solver.SetSolver(lin_solver)

        b = mfem.Vector()
        solver.Mult(b, x)
        iterations = solver.GetNumIterations()

        # x IS the mesh's nodes GF — mesh already updated in place.
        # Extract coordinates using DofToVDof (ordering-agnostic).
        n_nodes = mesh.GetNV()
        coords_adapted = np.zeros((n_nodes, dim))
        for i in range(n_nodes):
            for d in range(dim):
                dof = fes_mesh.DofToVDof(i, d)
                coords_adapted[i, d] = x[dof]

        return coords_adapted, iterations

    def _compute_mean_det(self, mesh) -> float:
        """
        Compute the mean absolute Jacobian determinant over all elements.

        Used to convert dimensionless size-scaling factors into physical
        units expected by DiscreteAdaptTC.SetSerialDiscreteTargetSize.
        """
        import mfem.ser as mfem

        n_elements = mesh.GetNE()
        if n_elements == 0:
            return 1.0

        total = 0.0
        for e in range(n_elements):
            T = mesh.GetElementTransformation(e)
            T.SetIntPoint(mfem.Geometries.GetCenter(mesh.GetElementGeometry(e)))
            total += abs(T.Jacobian().Det())
        return total / n_elements

    def _compute_quality(self, mesh) -> Dict[str, float]:
        """Compute mesh quality metrics."""
        import mfem.ser as mfem

        n_elements = mesh.GetNE()
        if n_elements == 0:
            return {"min_quality": 0.0, "avg_quality": 0.0, "num_inverted": 0}

        qualities = []
        num_inverted = 0

        for e in range(n_elements):
            T = mesh.GetElementTransformation(e)
            T.SetIntPoint(mfem.Geometries.GetCenter(mesh.GetElementGeometry(e)))
            det = T.Jacobian().Det()

            if det <= 0:
                num_inverted += 1
                qualities.append(0.0)
            else:
                qualities.append(min(det, 1.0 / det) if det != 0 else 0.0)

        qualities = np.array(qualities)

        return {
            "min_quality": float(np.min(qualities)),
            "avg_quality": float(np.mean(qualities)),
            "max_quality": float(np.max(qualities)),
            "num_inverted": num_inverted,
            "num_elements": n_elements,
        }


def is_tmop_available() -> bool:
    """
    Check if PyMFEM with TMOP support is available.

    Returns:
        True if TMOP can be used, False otherwise
    """
    try:
        import mfem.ser as mfem
        tmop = mfem.tmop
        return (
            hasattr(tmop, 'TMOP_Integrator') and
            hasattr(tmop, 'DiscreteAdaptTC') and
            hasattr(tmop, 'TMOP_Metric_002')
        )
    except (ImportError, AttributeError):
        return False
