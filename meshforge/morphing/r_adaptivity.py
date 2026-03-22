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
        barrier_type: Barrier function type ("shifted", "pseudo")
        max_iterations: Maximum Newton iterations
        tolerance: Newton solver tolerance
        min_det_threshold: Minimum Jacobian determinant (for barrier activation)
        verbosity: Output verbosity level
    """
    # Size scaling based on error
    size_scale_min: float = 0.3  # Target size in high-error regions (attracts nodes)
    size_scale_max: float = 2.0  # Target size in low-error regions (repels nodes)
    error_threshold: float = 0.1  # Normalize errors relative to max

    # Error field processing
    smoothing_iterations: int = 2  # Smooth error field to avoid oscillations

    # TMOP settings
    target_type: str = "ideal_shape_equal_size"
    quality_metric: str = "shape_and_size"
    barrier_type: str = "shifted"  # "shifted" or "pseudo" barrier
    max_iterations: int = 200
    tolerance: float = 1e-8
    min_det_threshold: float = 0.001  # Jacobian det threshold for barrier
    verbosity: int = 0

    # Boundary handling
    fix_boundary: bool = True  # Fix boundary nodes during adaptation


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
        self._solver = None
        self._mesh = None

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
            # Import PyMFEM
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

            # Validate error field
            if len(error_field) != n_nodes:
                return AdaptivityResult(
                    success=False,
                    coords_adapted=coords.copy(),
                    error_message=f"Error field length ({len(error_field)}) != "
                                 f"number of nodes ({n_nodes})"
                )

            # Get mesh from manager
            from ..mesh.mfem_manager import MFEMManager
            if not isinstance(manager, MFEMManager):
                return AdaptivityResult(
                    success=False,
                    coords_adapted=coords.copy(),
                    error_message="TMOPAdaptivity requires MFEMManager"
                )

            mesh = manager.mesh
            dim = mesh.Dimension()

            # Compute quality before
            quality_before = self._compute_quality(mesh)

            # Process error field
            processed_error = self._process_error_field(error_field, coords, mesh)

            # Compute target size field from error
            target_sizes = self._error_to_target_size(processed_error)

            # Create target size coefficient
            target_size_coeff = self._create_size_coefficient(
                mesh, target_sizes, manager.get_node_ids()
            )

            # Determine fixed nodes
            if fixed_nodes is None and self.config.fix_boundary:
                fixed_nodes = self._get_boundary_nodes(mesh)

            # Run TMOP optimization
            coords_adapted, iterations = self._run_tmop(
                mesh, target_size_coeff, fixed_nodes
            )

            # Update mesh
            manager.update_nodes(coords_adapted)

            # Compute quality after
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
        """
        Process error field: normalize and smooth.

        Args:
            error_field: Raw error values at nodes
            coords: Node coordinates
            mesh: MFEM mesh

        Returns:
            Processed error field (normalized to [0, 1])
        """
        # Ensure non-negative
        error = np.maximum(error_field, 0.0)

        # Normalize to [0, 1]
        max_error = np.max(error)
        if max_error > 1e-12:
            error = error / max_error
        else:
            # Uniform error - no adaptation needed
            error = np.ones_like(error) * 0.5

        # Apply Laplacian smoothing to avoid oscillations
        for _ in range(self.config.smoothing_iterations):
            error = self._laplacian_smooth(error, coords, mesh)

        return error

    def _laplacian_smooth(
        self,
        field: np.ndarray,
        coords: np.ndarray,
        mesh
    ) -> np.ndarray:
        """
        Apply one iteration of Laplacian smoothing.

        Args:
            field: Field values at nodes
            coords: Node coordinates
            mesh: MFEM mesh

        Returns:
            Smoothed field
        """
        import mfem.ser as mfem

        n_nodes = len(field)
        smoothed = field.copy()

        # Build node adjacency from elements
        # For each node, average with its neighbors
        neighbor_sum = np.zeros(n_nodes)
        neighbor_count = np.zeros(n_nodes)

        n_elements = mesh.GetNE()
        for e in range(n_elements):
            elem = mesh.GetElement(e)
            vertices = elem.GetVerticesArray()

            # Each vertex is neighbor to all others in element
            for v in vertices:
                for u in vertices:
                    if u != v:
                        neighbor_sum[v] += field[u]
                        neighbor_count[v] += 1

        # Apply smoothing where we have neighbors
        mask = neighbor_count > 0
        smoothed[mask] = 0.5 * field[mask] + 0.5 * neighbor_sum[mask] / neighbor_count[mask]

        return smoothed

    def _error_to_target_size(self, error: np.ndarray) -> np.ndarray:
        """
        Convert normalized error field to target element size field.

        High error → small target size (attracts nodes)
        Low error → large target size (repels nodes)

        Args:
            error: Normalized error field [0, 1]

        Returns:
            Target size scaling factors
        """
        # Linear interpolation between size_scale_min and size_scale_max
        # High error (1) → size_scale_min
        # Low error (0) → size_scale_max
        size_min = self.config.size_scale_min
        size_max = self.config.size_scale_max

        # Invert: high error → small size
        target_sizes = size_max - (size_max - size_min) * error

        return target_sizes

    def _create_size_coefficient(
        self,
        mesh,
        target_sizes: np.ndarray,
        node_ids: np.ndarray
    ):
        """
        Create MFEM coefficient for target element sizes.

        Args:
            mesh: MFEM mesh
            target_sizes: Target size at each node
            node_ids: Node IDs

        Returns:
            MFEM Coefficient for target size field
        """
        import mfem.ser as mfem

        # Create a GridFunction to hold the target size field
        # We'll use H1 finite element space (continuous)
        fec = mfem.H1_FECollection(1, mesh.Dimension())
        fes = mfem.FiniteElementSpace(mesh, fec)

        # Create grid function and set nodal values
        size_gf = mfem.GridFunction(fes)

        # Map target sizes to the grid function
        # GridFunction uses same node ordering as mesh vertices
        for i, node_id in enumerate(node_ids):
            if i < size_gf.Size():
                size_gf[i] = target_sizes[i]

        # Create coefficient from grid function
        return mfem.GridFunctionCoefficient(size_gf), fec, fes, size_gf

    def _get_boundary_nodes(self, mesh) -> np.ndarray:
        """
        Get indices of boundary nodes.

        Args:
            mesh: MFEM mesh

        Returns:
            Array of boundary node indices
        """
        import mfem.ser as mfem

        boundary_nodes = set()

        # Iterate over boundary elements
        n_bdr = mesh.GetNBE()
        for b in range(n_bdr):
            bdr_elem = mesh.GetBdrElement(b)
            vertices = bdr_elem.GetVerticesArray()
            boundary_nodes.update(vertices)

        return np.array(list(boundary_nodes), dtype=int)

    def _run_tmop(
        self,
        mesh,
        target_size_coeff,
        fixed_nodes: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """
        Run TMOP optimization with barrier functions.

        Args:
            mesh: MFEM mesh
            target_size_coeff: Tuple of (coefficient, fec, fes, gf) for target size
            fixed_nodes: Indices of nodes to keep fixed

        Returns:
            Tuple of (adapted coordinates, number of iterations)
        """
        import mfem.ser as mfem

        dim = mesh.Dimension()
        coeff, fec, fes, size_gf = target_size_coeff

        # Create finite element space for mesh coordinates
        fec_mesh = mfem.H1_FECollection(1, dim)
        fes_mesh = mfem.FiniteElementSpace(mesh, fec_mesh, dim)

        # Get initial mesh node coordinates as GridFunction
        x = mfem.GridFunction(fes_mesh)
        mesh.GetNodes(x)

        # Setup target construction
        # Use target type that respects size specification
        target_c = mfem.TargetConstructor(
            mfem.TargetConstructor.IDEAL_SHAPE_GIVEN_SIZE
        )
        target_c.SetNodes(x)

        # Set the target size field
        # This makes TMOP optimize toward these target element sizes
        target_c.SetSizeGF(size_gf)

        # Choose TMOP metric based on config
        if dim == 2:
            # 2D metrics
            if self.config.quality_metric == "shape":
                mu = mfem.TMOP_Metric_002()  # Shape metric
            elif self.config.quality_metric == "size":
                mu = mfem.TMOP_Metric_080()  # Size metric
            else:  # shape_and_size
                mu = mfem.TMOP_Metric_077()  # Combined shape+size
        else:
            # 3D metrics
            if self.config.quality_metric == "shape":
                mu = mfem.TMOP_Metric_302()
            elif self.config.quality_metric == "size":
                mu = mfem.TMOP_Metric_316()
            else:
                mu = mfem.TMOP_Metric_321()

        # Setup barrier for preventing element inversion
        if self.config.barrier_type == "shifted":
            # Shifted barrier - more aggressive
            mu.SetBarrierType(mfem.TMOP_QualityMetric.SHIFTED_BARRIER)
        else:
            # Pseudo barrier - smoother
            mu.SetBarrierType(mfem.TMOP_QualityMetric.PSEUDO_BARRIER)

        mu.SetMinDetT(self.config.min_det_threshold)

        # Create TMOP integrator
        tmop_integ = mfem.TMOP_Integrator(mu, target_c)

        # Enable finite difference for gradient/Hessian if needed
        tmop_integ.EnableFiniteDifferences(x)

        # Limiting to prevent large node movements (optional stabilization)
        tmop_integ.EnableLimiting(x, mfem.ConstantCoefficient(1.0))

        # Create nonlinear form
        a = mfem.NonlinearForm(fes_mesh)
        a.AddDomainIntegrator(tmop_integ)

        # Setup essential boundary conditions for fixed nodes
        if fixed_nodes is not None and len(fixed_nodes) > 0:
            ess_tdof_list = mfem.intArray()
            # Convert node indices to true DOF indices
            for node_idx in fixed_nodes:
                for d in range(dim):
                    dof = node_idx * dim + d
                    if dof < fes_mesh.GetTrueVSize():
                        ess_tdof_list.Append(dof)
            a.SetEssentialTrueDofs(ess_tdof_list)

        # Newton solver
        solver = mfem.NewtonSolver()
        solver.SetMaxIter(self.config.max_iterations)
        solver.SetRelTol(self.config.tolerance)
        solver.SetAbsTol(0.0)
        solver.SetPrintLevel(self.config.verbosity)

        # Linear solver for Newton steps
        linear_solver = mfem.UMFPackSolver()
        solver.SetSolver(linear_solver)
        solver.SetOperator(a)

        # Solve
        b = mfem.Vector()  # Empty RHS
        solver.Mult(b, x)

        iterations = solver.GetNumIterations()

        # Update mesh with new coordinates
        mesh.SetNodes(x)

        # Extract coordinates as numpy array
        n_nodes = mesh.GetNV()
        coords_adapted = np.zeros((n_nodes, dim))
        for i in range(n_nodes):
            v = mesh.GetVertex(i)
            for d in range(dim):
                coords_adapted[i, d] = v[d]

        return coords_adapted, iterations

    def _compute_quality(self, mesh) -> Dict[str, float]:
        """
        Compute mesh quality metrics.

        Args:
            mesh: MFEM mesh

        Returns:
            Dictionary of quality metrics
        """
        import mfem.ser as mfem

        n_elements = mesh.GetNE()
        if n_elements == 0:
            return {"min_quality": 0.0, "avg_quality": 0.0, "num_inverted": 0}

        qualities = []
        num_inverted = 0

        for e in range(n_elements):
            # Get element transformation
            T = mesh.GetElementTransformation(e)
            T.SetIntPoint(mfem.Geometries.GetCenter(mesh.GetElementGeometry(e)))

            # Jacobian determinant
            det = T.Jacobian().Det()

            if det <= 0:
                num_inverted += 1
                qualities.append(0.0)
            else:
                # Simple quality metric: based on Jacobian condition
                # More sophisticated metrics could be added
                qualities.append(min(det, 1.0 / det) if det != 0 else 0.0)

        qualities = np.array(qualities)

        return {
            "min_quality": float(np.min(qualities)) if len(qualities) > 0 else 0.0,
            "avg_quality": float(np.mean(qualities)) if len(qualities) > 0 else 0.0,
            "max_quality": float(np.max(qualities)) if len(qualities) > 0 else 0.0,
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
        # Check for TMOP classes
        return (
            hasattr(mfem, 'TMOP_Integrator') and
            hasattr(mfem, 'TargetConstructor') and
            hasattr(mfem, 'TMOP_Metric_002')
        )
    except ImportError:
        return False
