"""
Phase field fracture solver using DOLFINx.

Implements the AT2 (Ambrosio-Tortorelli) phase field model for brittle fracture
with a staggered solution scheme.

References:
- Bourdin, B., Francfort, G. A., & Marigo, J. J. (2000). Numerical experiments
  in revisited brittle fracture. Journal of the Mechanics and Physics of Solids.
- Miehe, C., Hofacker, M., & Welschinger, F. (2010). A phase field model for
  rate-independent crack propagation. Computer Methods in Applied Mechanics
  and Engineering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

HAS_DOLFINX = False

try:
    from mpi4py import MPI
    import dolfinx
    from dolfinx import fem, default_scalar_type
    from dolfinx.fem import Function, FunctionSpace, dirichletbc, form, locate_dofs_topological
    from dolfinx.fem.petsc import LinearProblem, assemble_vector, assemble_matrix, create_vector
    from dolfinx.mesh import locate_entities_boundary, meshtags
    from dolfinx.io import XDMFFile
    import ufl
    from ufl import (
        TestFunction, TrialFunction, dx, grad, inner, dot, div, sym,
        tr, Identity, variable, diff, conditional, gt, sqrt
    )
    from petsc4py import PETSc
    HAS_DOLFINX = True
except ImportError:
    pass

if TYPE_CHECKING:
    from dolfinx.fem import Function, FunctionSpace

from .base import (
    SolverInterface,
    SolverResult,
    PhysicsConfig,
    PhysicsType,
    PhaseFieldConfig,
    MaterialProperties,
)
from ..mesh.fenics_manager import FEniCSManager


@dataclass
class PhaseFieldState:
    """
    State variables for phase field solver.

    Attributes:
        displacement: Current displacement field
        damage: Current damage (phase) field
        damage_prev: Damage from previous load step (for irreversibility)
        history: History field H = max(psi_0, H_prev)
        load_factor: Current load factor
    """
    displacement: "Function"
    damage: "Function"
    damage_prev: "Function"
    history: "Function"
    load_factor: float = 0.0


class FEniCSPhaseFieldSolver(SolverInterface):
    """
    Phase field fracture solver using DOLFINx.

    Implements the AT2 model with staggered (alternate minimization) scheme:
    1. Solve displacement problem (fixed damage)
    2. Update history field H = max(psi_0, H_prev)
    3. Solve damage problem (fixed displacement, irreversibility)
    4. Check convergence, iterate if needed

    The total energy functional:
        E = ∫ g(d)·ψ₀(ε) dx + ∫ G_c·(d²/(2l) + l/2·|∇d|²) dx

    where:
        g(d) = (1-d)² + k_res  (degradation function)
        ψ₀(ε) = λ/2·(tr ε)² + μ·(ε:ε)  (elastic strain energy)
    """

    def __init__(self):
        """Initialize the solver."""
        super().__init__()

        if not HAS_DOLFINX:
            raise ImportError(
                "DOLFINx is required for FEniCSPhaseFieldSolver. "
                "Install with: conda install -c conda-forge fenics-dolfinx"
            )

        self._state: Optional[PhaseFieldState] = None
        self._V: Optional[FunctionSpace] = None  # Vector space for displacement
        self._W: Optional[FunctionSpace] = None  # Scalar space for damage
        self._results: List[Dict] = []

    def setup(
        self,
        mesh_manager: FEniCSManager,
        physics: PhysicsConfig,
    ) -> None:
        """
        Set up the solver with mesh and physics configuration.

        Args:
            mesh_manager: FEniCS mesh manager instance
            physics: Physics configuration with phase field settings
        """
        super().setup(mesh_manager, physics)

        if physics.physics_type != PhysicsType.PHASE_FIELD_FRACTURE:
            raise ValueError(
                f"Expected PHASE_FIELD_FRACTURE physics, got {physics.physics_type}"
            )

        if physics.phase_field is None:
            raise ValueError("PhaseFieldConfig must be provided for phase field solver")

        mesh = mesh_manager.mesh
        self._pf_config = physics.phase_field
        self._material = physics.material

        # Create function spaces
        self._V = fem.functionspace(mesh, ("Lagrange", 1, (mesh_manager.dimension,)))
        self._W = fem.functionspace(mesh, ("Lagrange", 1))
        self._DG0 = fem.functionspace(mesh, ("DG", 0))

        # Initialize state
        self._state = PhaseFieldState(
            displacement=Function(self._V, name="displacement"),
            damage=Function(self._W, name="damage"),
            damage_prev=Function(self._W, name="damage_prev"),
            history=Function(self._DG0, name="history"),
            load_factor=0.0,
        )

        # Set up variational forms
        self._setup_displacement_problem()
        self._setup_damage_problem()

        self._is_setup = True

    def _setup_displacement_problem(self) -> None:
        """Set up the displacement variational problem."""
        u = TrialFunction(self._V)
        v = TestFunction(self._V)
        d = self._state.damage

        # Material parameters
        E = self._material.E
        nu = self._material.nu
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        # Phase field parameters
        k_res = self._pf_config.k_res

        # Degradation function g(d) = (1-d)^2 + k_res
        g_d = (1 - d) ** 2 + k_res

        # Strain and stress
        def epsilon(u):
            return sym(grad(u))

        def sigma(u, g):
            eps = epsilon(u)
            return g * (lmbda * tr(eps) * Identity(len(u)) + 2 * mu * eps)

        # Bilinear and linear forms
        self._a_u = form(inner(sigma(u, g_d), epsilon(v)) * dx)
        self._L_u = form(inner(fem.Constant(self._mesh_manager.mesh, default_scalar_type((0.0,) * self._mesh_manager.dimension)), v) * dx)

        # Store for later use
        self._epsilon = epsilon
        self._lmbda = lmbda
        self._mu = mu

    def _setup_damage_problem(self) -> None:
        """Set up the damage variational problem."""
        d = TrialFunction(self._W)
        q = TestFunction(self._W)
        H = self._state.history

        # Phase field parameters
        G_c = self._pf_config.G_c
        l_0 = self._pf_config.l_0

        # Bilinear form
        self._a_d = form(
            (G_c * l_0 * inner(grad(d), grad(q)) + (G_c / l_0 + 2 * H) * d * q) * dx
        )

        # Linear form
        self._L_d = form(2 * H * q * dx)

    def _compute_strain_energy(self, u: Function) -> Function:
        """Compute elastic strain energy density ψ₀(ε)."""
        eps = self._epsilon(u)
        psi = 0.5 * self._lmbda * tr(eps) ** 2 + self._mu * inner(eps, eps)

        # Project to DG0
        psi_proj = Function(self._DG0)
        psi_expr = fem.Expression(psi, self._DG0.element.interpolation_points())
        psi_proj.interpolate(psi_expr)

        return psi_proj

    def _update_history(self) -> None:
        """Update history field H = max(ψ₀, H_prev)."""
        psi = self._compute_strain_energy(self._state.displacement)

        # H = max(psi, H_prev)
        H_vals = self._state.history.x.array
        psi_vals = psi.x.array
        np.maximum(H_vals, psi_vals, out=H_vals)

    def _apply_boundary_conditions(
        self,
        load_factor: float,
    ) -> Tuple[List, List]:
        """
        Apply boundary conditions for current load step.

        Returns displacement and damage BCs.
        """
        mesh = self._mesh_manager.mesh
        V = self._V
        W = self._W
        dim = self._mesh_manager.dimension

        bc_u = []
        bc_d = []

        # Process boundary conditions from physics config
        for bc in self._physics.boundary_conditions:
            marker = bc.boundary_id

            # Get facets for this boundary
            if self._mesh_manager.facet_markers is not None:
                facets = self._mesh_manager.facet_markers.find(marker)
            else:
                # Fall back to geometric location
                facets = self._locate_boundary_facets(marker)

            if bc.bc_type.name == "DISPLACEMENT":
                if bc.value is not None:
                    value = np.array(bc.value) * load_factor if np.isscalar(bc.value) or len(bc.value) > 1 else bc.value * load_factor
                else:
                    value = np.zeros(dim)

                # Create BC
                dofs = locate_dofs_topological(V, dim - 1, facets)
                u_bc = Function(V)
                u_bc.x.array[:] = 0.0
                # Set boundary value
                bc_u.append(dirichletbc(u_bc, dofs))

            elif bc.bc_type.name == "TRACTION":
                # Traction BCs handled in weak form
                pass

        # Damage BC: d = 0 on all boundaries (no pre-existing damage at boundaries)
        # This is optional and depends on problem setup

        return bc_u, bc_d

    def _locate_boundary_facets(self, marker: int) -> np.ndarray:
        """Locate boundary facets by marker using geometric criteria."""
        mesh = self._mesh_manager.mesh
        dim = mesh.topology.dim

        # Define boundary locators based on marker convention
        # 1=bottom, 2=right, 3=top, 4=left
        W = self._mesh_manager.mesh.geometry.x[:, 0].max()
        H = self._mesh_manager.mesh.geometry.x[:, 1].max()
        eps = 1e-6

        def bottom(x):
            return np.isclose(x[1], 0, atol=eps)

        def right(x):
            return np.isclose(x[0], W, atol=eps)

        def top(x):
            return np.isclose(x[1], H, atol=eps)

        def left(x):
            return np.isclose(x[0], 0, atol=eps)

        locators = {1: bottom, 2: right, 3: top, 4: left}

        if marker in locators:
            facets = locate_entities_boundary(mesh, dim - 1, locators[marker])
            return facets
        else:
            return np.array([], dtype=np.int32)

    def _solve_displacement(self, bc_u: List) -> float:
        """
        Solve displacement subproblem.

        Returns residual norm.
        """
        # Reassemble with current damage
        self._setup_displacement_problem()

        problem = LinearProblem(
            self._a_u, self._L_u, bcs=bc_u,
            petsc_options={"ksp_type": "cg", "pc_type": "hypre"}
        )
        self._state.displacement = problem.solve()

        return problem.solver.getConvergedReason()

    def _solve_damage(self, bc_d: List) -> float:
        """
        Solve damage subproblem with irreversibility constraint.

        Returns residual norm.
        """
        problem = LinearProblem(
            self._a_d, self._L_d, bcs=bc_d,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi"}
        )
        d_new = problem.solve()

        # Enforce irreversibility: d >= d_prev
        d_vals = d_new.x.array
        d_prev_vals = self._state.damage_prev.x.array
        np.maximum(d_vals, d_prev_vals, out=d_vals)

        # Enforce bounds: 0 <= d <= 1
        np.clip(d_vals, 0.0, 1.0, out=d_vals)

        # Update damage
        self._state.damage.x.array[:] = d_vals

        return problem.solver.getConvergedReason()

    def _staggered_iteration(
        self,
        load_factor: float,
    ) -> Tuple[bool, int]:
        """
        Perform staggered iteration for one load step.

        Returns (converged, n_iterations).
        """
        bc_u, bc_d = self._apply_boundary_conditions(load_factor)

        d_old = self._state.damage.x.array.copy()

        for iteration in range(self._pf_config.stagger_max_iter):
            # Solve displacement
            self._solve_displacement(bc_u)

            # Update history field
            self._update_history()

            # Solve damage
            self._solve_damage(bc_d)

            # Check convergence
            d_new = self._state.damage.x.array
            diff = np.linalg.norm(d_new - d_old)
            d_old = d_new.copy()

            if diff < self._pf_config.stagger_tol:
                return True, iteration + 1

        return False, self._pf_config.stagger_max_iter

    def solve(self, output_dir: Union[str, Path]) -> SolverResult:
        """
        Execute the phase field fracture solver.

        Performs quasi-static load stepping with staggered scheme.

        Args:
            output_dir: Directory to store output files

        Returns:
            SolverResult with solution data
        """
        if not self._is_setup:
            raise RuntimeError("Solver not set up. Call setup() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import time
        start_time = time.time()

        n_steps = self._pf_config.n_load_steps
        self._results = []

        # Load stepping
        for step in range(n_steps):
            load_factor = (step + 1) / n_steps
            self._state.load_factor = load_factor

            converged, n_iter = self._staggered_iteration(load_factor)

            if not converged:
                return SolverResult(
                    success=False,
                    error_message=f"Staggered scheme did not converge at step {step}",
                    solve_time=time.time() - start_time,
                )

            # Update previous damage for irreversibility
            self._state.damage_prev.x.array[:] = self._state.damage.x.array

            # Store results for this step
            self._results.append({
                "step": step,
                "load_factor": load_factor,
                "n_iterations": n_iter,
                "max_damage": float(self._state.damage.x.array.max()),
            })

        # Extract final solution
        solution_data = {
            "displacement": self._state.displacement.x.array.reshape(-1, self._mesh_manager.dimension).copy(),
            "damage": self._state.damage.x.array.copy(),
        }

        # Extract crack path (where d > threshold)
        crack_path = self._extract_crack_path()
        if crack_path is not None:
            solution_data["crack_path"] = crack_path

        # Compute von Mises stress
        von_mises = self._compute_von_mises()
        if von_mises is not None:
            solution_data["von_mises"] = von_mises

        # Save output
        output_file = output_dir / "phase_field_result.xdmf"
        self._save_results(output_file)

        return SolverResult(
            success=True,
            solution_data=solution_data,
            output_files=[output_file],
            metrics={
                "n_load_steps": n_steps,
                "max_damage": float(self._state.damage.x.array.max()),
                "results_per_step": self._results,
            },
            solve_time=time.time() - start_time,
        )

    def _extract_crack_path(self) -> Optional[np.ndarray]:
        """
        Extract crack path from damage field.

        Returns coordinates where d > damage_threshold.
        """
        threshold = self._pf_config.damage_threshold
        d_vals = self._state.damage.x.array
        coords = self._mesh_manager.get_nodes()

        # DOF to node mapping for Lagrange elements
        mask = d_vals > threshold

        if not mask.any():
            return None

        crack_coords = coords[mask]

        # Sort by x-coordinate for path representation
        sort_idx = np.argsort(crack_coords[:, 0])
        return crack_coords[sort_idx]

    def _compute_von_mises(self) -> Optional[np.ndarray]:
        """Compute von Mises stress at nodes."""
        u = self._state.displacement
        d = self._state.damage

        # Degraded stress
        E = self._material.E
        nu = self._material.nu
        lmbda = self._lmbda
        mu = self._mu
        k_res = self._pf_config.k_res

        eps = self._epsilon(u)
        g_d = (1 - d) ** 2 + k_res
        sigma = g_d * (lmbda * tr(eps) * Identity(self._mesh_manager.dimension) + 2 * mu * eps)

        # Von Mises: sqrt(3/2 * s:s) where s = sigma - 1/3*tr(sigma)*I
        if self._mesh_manager.dimension == 2:
            # Plane stress approximation
            s = sigma - (1.0 / 3.0) * tr(sigma) * Identity(2)
            von_mises_expr = sqrt(1.5 * inner(s, s))
        else:
            s = sigma - (1.0 / 3.0) * tr(sigma) * Identity(3)
            von_mises_expr = sqrt(1.5 * inner(s, s))

        # Project to function space
        V_scalar = fem.functionspace(self._mesh_manager.mesh, ("Lagrange", 1))
        von_mises_func = Function(V_scalar)

        try:
            expr = fem.Expression(von_mises_expr, V_scalar.element.interpolation_points())
            von_mises_func.interpolate(expr)
            return von_mises_func.x.array.copy()
        except Exception:
            return None

    def _save_results(self, output_path: Path) -> None:
        """Save results to XDMF file."""
        with XDMFFile(MPI.COMM_WORLD, str(output_path), "w") as xdmf:
            xdmf.write_mesh(self._mesh_manager.mesh)
            xdmf.write_function(self._state.displacement)
            xdmf.write_function(self._state.damage)

    def get_solution_field(self, field_name: str) -> np.ndarray:
        """
        Extract a solution field.

        Args:
            field_name: Name of the field ("displacement", "damage", "von_mises")

        Returns:
            np.ndarray: Solution field data
        """
        if not self._is_setup or self._state is None:
            raise RuntimeError("No solution available. Call solve() first.")

        if field_name == "displacement":
            return self._state.displacement.x.array.reshape(-1, self._mesh_manager.dimension)
        elif field_name == "damage":
            return self._state.damage.x.array
        elif field_name == "von_mises":
            return self._compute_von_mises()
        else:
            raise ValueError(f"Unknown field: {field_name}")

    def get_available_fields(self) -> List[str]:
        """Get list of available solution fields."""
        return ["displacement", "damage", "von_mises", "crack_path"]

    @property
    def state(self) -> Optional[PhaseFieldState]:
        """Get current solver state."""
        return self._state
