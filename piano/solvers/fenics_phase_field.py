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
    from dolfinx.fem import Function, FunctionSpace, dirichletbc, form, locate_dofs_topological, locate_dofs_geometrical
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

        # Create function spaces — dolfinx 0.9+ requires basix.ufl.element for vector spaces
        import basix.ufl as bufl
        gdim = mesh_manager.dimension
        vec_elem = bufl.element("Lagrange", mesh.basix_cell(), 1, shape=(gdim,))
        self._V   = fem.functionspace(mesh, vec_elem)
        self._W   = fem.functionspace(mesh, ("Lagrange", 1))
        self._DG0 = fem.functionspace(mesh, ("DG", 0))

        # Initialize state
        self._state = PhaseFieldState(
            displacement=Function(self._V, name="displacement"),
            damage=Function(self._W, name="damage"),
            damage_prev=Function(self._W, name="damage_prev"),
            history=Function(self._DG0, name="history"),
            load_factor=0.0,
        )

        # Build traction constants for each Neumann BC (updated each load step)
        self._traction_constants: dict = {}
        for bc_cfg in physics.boundary_conditions:
            if bc_cfg.bc_type.name == "TRACTION" and bc_cfg.value is not None:
                trac_full = np.array(bc_cfg.value, dtype=np.float64)
                tc = fem.Constant(mesh, default_scalar_type(tuple(np.zeros_like(trac_full))))
                self._traction_constants[bc_cfg.boundary_id] = (tc, trac_full)

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

        # Store raw UFL forms — LinearProblem compiles them internally
        self._a_u = inner(sigma(u, g_d), epsilon(v)) * dx

        # Body force (zero) + Neumann traction on each marked boundary
        mesh = self._mesh_manager.mesh
        dim = self._mesh_manager.dimension
        if self._mesh_manager.facet_markers is not None:
            ds_meas = ufl.Measure("ds", domain=mesh, subdomain_data=self._mesh_manager.facet_markers)
        else:
            ds_meas = ufl.Measure("ds", domain=mesh)
        L = inner(fem.Constant(mesh, default_scalar_type((0.0,) * dim)), v) * dx
        for marker, (tc, _) in self._traction_constants.items():
            L = L + inner(tc, v) * ds_meas(marker)
        self._L_u = L

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

        # Store raw UFL forms — LinearProblem compiles them internally
        self._a_d = (G_c * l_0 * inner(grad(d), grad(q)) + (G_c / l_0 + 2 * H) * d * q) * dx
        self._L_d = 2 * H * q * dx

    def _compute_strain_energy(self, u: Function) -> Function:
        """Compute elastic strain energy density ψ₀(ε)."""
        eps = self._epsilon(u)
        psi = 0.5 * self._lmbda * tr(eps) ** 2 + self._mu * inner(eps, eps)

        # Project to DG0
        psi_proj = Function(self._DG0)
        psi_expr = fem.Expression(psi, self._DG0.element.interpolation_points)
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
                if bc.direction is not None:
                    # Component-specific BC — apply only to sub-space bc.direction
                    comp = bc.direction
                    scalar_val = float(bc.value) * load_factor if bc.value is not None else 0.0
                    V_sub, sub_to_parent = V.sub(comp).collapse()
                    dofs_sub = locate_dofs_topological(
                        (V.sub(comp), V_sub), dim - 1, facets
                    )
                    u_sub = Function(V_sub)
                    u_sub.x.array[:] = scalar_val
                    bc_u.append(dirichletbc(u_sub, dofs_sub, V.sub(comp)))
                else:
                    # Full-vector BC (all components)
                    if bc.value is not None:
                        value = np.array(bc.value, dtype=float)
                    else:
                        value = np.zeros(dim)
                    dofs = locate_dofs_topological(V, dim - 1, facets)
                    u_bc = Function(V)
                    u_bc.x.array[:] = 0.0
                    bc_u.append(dirichletbc(u_bc, dofs))

            elif bc.bc_type.name == "TRACTION":
                # Traction BCs handled in weak form
                pass

        # Anti-rigid-body: fix u_x = 0 at bottom-left corner (0, 0).
        # Prevents the unconstrained x-translation mode without creating
        # corner stress singularities (single point, not a full boundary).
        V_x, _ = V.sub(0).collapse()
        dofs_corner = locate_dofs_geometrical(
            (V.sub(0), V_x),
            lambda x: np.isclose(x[0], 0.0, atol=1e-10) & np.isclose(x[1], 0.0, atol=1e-10),
        )
        if len(dofs_corner[0]) > 0:
            u_x_zero = Function(V_x)
            u_x_zero.x.array[:] = 0.0
            bc_u.append(dirichletbc(u_x_zero, dofs_corner, V.sub(0)))

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
        problem = LinearProblem(
            self._a_u, self._L_u, petsc_options_prefix="u_",
            bcs=bc_u, petsc_options={"ksp_type": "cg", "pc_type": "hypre"}
        )
        self._state.displacement = problem.solve()

        return problem.solver.getConvergedReason()

    def _solve_damage(self, bc_d: List) -> float:
        """
        Solve damage subproblem with irreversibility constraint.

        Returns residual norm.
        """
        problem = LinearProblem(
            self._a_d, self._L_d, petsc_options_prefix="d_",
            bcs=bc_d, petsc_options={"ksp_type": "cg", "pc_type": "jacobi"}
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

        # Scale traction constants for this load step
        for marker, (tc, full_value) in self._traction_constants.items():
            tc.value[:] = full_value * load_factor

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

    def _initialize_crack_damage(self) -> None:
        """Pre-set the AT-2 optimal damage profile on the pre-existing crack.

        The 1D AT-2 optimal profile is d(y) = exp(-|y - crack_y| / (2·l₀)).
        This is the exact minimiser of the AT-2 crack surface energy for a
        straight crack, so it starts the staggered scheme in a consistent state
        instead of the sharp d=1-only-on-crack-line seed that leaves most of the
        process zone un-initialised and weakens the stress concentration.

        Nodes with x > crack_tip_x + l_0/2 are left at d=0 (intact ahead of tip).
        """
        cfg = self._pf_config
        if cfg.crack_tip_x is None or cfg.crack_y is None:
            return

        dof_coords = self._W.tabulate_dof_coordinates()
        x_dof = dof_coords[:, 0]
        y_dof = dof_coords[:, 1]

        # Only initialise nodes that are on the crack body (not ahead of tip)
        tol_x = cfg.l_0 * 0.5
        in_crack_region = x_dof <= cfg.crack_tip_x + tol_x

        # AT-2 optimal 1D profile: d = exp(-|y - crack_y| / (2·l₀))
        dist_y = np.abs(y_dof - cfg.crack_y)
        d_profile = np.exp(-dist_y / (2.0 * cfg.l_0))

        # Only apply where the profile is non-trivial (within ~5·l₀ of crack)
        active = in_crack_region & (d_profile > 0.01)
        n_active = int(active.sum())

        if n_active == 0:
            return

        # Take the maximum so we never reduce existing damage (irreversibility)
        d_arr = self._state.damage.x.array
        d_prev_arr = self._state.damage_prev.x.array
        d_arr[active] = np.maximum(d_arr[active], d_profile[active])
        d_prev_arr[active] = np.maximum(d_prev_arr[active], d_profile[active])
        self._state.damage.x.scatter_forward()
        self._state.damage_prev.x.scatter_forward()

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

        # Pre-initialize d=1 on the pre-existing crack faces (standard AT-2 practice).
        # Without this, the mesh notch is zero-width and provides no stress concentration.
        self._initialize_crack_damage()

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
            "elements": self._mesh_manager.get_elements(),
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

        # Von Mises: sqrt(3/2 * s:s) where s = dev(sigma) = sigma - 1/3*tr(sigma)*I
        dim = self._mesh_manager.dimension
        s = sigma - (1.0 / 3.0) * tr(sigma) * Identity(dim)
        von_mises_expr = sqrt(1.5 * inner(s, s))

        # Project onto CG1 scalar space — interpolation_points is a property in dolfinx 0.10
        V_scalar = fem.functionspace(self._mesh_manager.mesh, ("Lagrange", 1))
        von_mises_func = Function(V_scalar)

        try:
            expr = fem.Expression(von_mises_expr, V_scalar.element.interpolation_points)
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
