"""
Abstract base class and dataclasses for FEM solvers.

Provides a unified interface for solver operations across different FEM backends
(Abaqus, MFEM, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..mesh.base import MeshManager


class PhysicsType(Enum):
    """Enumeration of supported physics types."""
    LINEAR_ELASTICITY = auto()
    HEAT_TRANSFER = auto()
    PHASE_FIELD_FRACTURE = auto()
    # Future
    INCOMPRESSIBLE_NAVIER_STOKES = auto()
    STOKES = auto()
    NONLINEAR_ELASTICITY = auto()


class BoundaryConditionType(Enum):
    """Enumeration of boundary condition types."""
    # Mechanical
    DISPLACEMENT = auto()
    TRACTION = auto()
    SYMMETRY = auto()
    # Thermal
    TEMPERATURE = auto()
    HEAT_FLUX = auto()
    CONVECTION = auto()
    # Fluid (future)
    VELOCITY = auto()
    PRESSURE = auto()
    SLIP = auto()
    PERIODIC = auto()


@dataclass
class FluidProperties:
    """
    Material properties for fluid problems (Navier-Stokes / Stokes).

    Attributes:
        dynamic_viscosity: μ (Pa·s)
        density: ρ (kg/m³)
    """
    dynamic_viscosity: float  # μ (Pa·s)
    density: float            # ρ (kg/m³)

    def __post_init__(self):
        if self.dynamic_viscosity <= 0:
            raise ValueError("dynamic_viscosity must be positive")
        if self.density <= 0:
            raise ValueError("density must be positive")

    @property
    def kinematic_viscosity(self) -> float:
        """ν = μ / ρ (m²/s)"""
        return self.dynamic_viscosity / self.density


@dataclass
class MaterialProperties:
    """
    Material properties for FEM analysis.

    Attributes:
        E: Young's modulus (Pa)
        nu: Poisson's ratio
        density: Mass density (kg/m^3)
        k: Thermal conductivity (W/(m*K))
        cp: Specific heat capacity (J/(kg*K))
        fluid: Fluid properties for Navier-Stokes / Stokes problems (optional)
    """
    E: float = 200e9
    nu: float = 0.3
    density: float = 7850.0
    k: float = 50.0
    cp: float = 500.0
    fluid: Optional[FluidProperties] = None

    def __post_init__(self):
        """Validate material properties."""
        if self.E <= 0:
            raise ValueError("Young's modulus E must be positive")
        if not -1.0 < self.nu < 0.5:
            raise ValueError("Poisson's ratio nu must be in range (-1, 0.5)")
        if self.density <= 0:
            raise ValueError("Density must be positive")
        if self.k <= 0:
            raise ValueError("Thermal conductivity k must be positive")
        if self.cp <= 0:
            raise ValueError("Specific heat capacity cp must be positive")

    @property
    def lame_lambda(self) -> float:
        """Calculate Lame's first parameter."""
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def lame_mu(self) -> float:
        """Calculate Lame's second parameter (shear modulus)."""
        return self.E / (2 * (1 + self.nu))

    @property
    def bulk_modulus(self) -> float:
        """Calculate bulk modulus."""
        return self.E / (3 * (1 - 2 * self.nu))


@dataclass
class BoundaryCondition:
    """
    Boundary condition configuration.

    Attributes:
        bc_type: Type of boundary condition
        boundary_id: Boundary attribute/ID in the mesh
        value: BC value (scalar, vector, or None for symmetry/slip)
        direction: Direction for displacement BCs (0=x, 1=y, 2=z)
    """
    bc_type: BoundaryConditionType
    boundary_id: int
    value: Optional[Union[float, np.ndarray]] = None
    direction: Optional[int] = None

    def __post_init__(self):
        """Validate boundary condition."""
        if self.bc_type == BoundaryConditionType.DISPLACEMENT:
            if self.value is None and self.direction is None:
                raise ValueError(
                    "Displacement BC requires either value or direction"
                )


@dataclass
class TransientConfig:
    """
    Time integration settings.

    When set on PhysicsConfig, the solver runs a time-stepping loop.
    None (default) means steady-state — existing behaviour is unchanged.

    Attributes:
        t_start: Start time
        t_end: End time
        dt: Time step size
        scheme: Time integration scheme
        save_interval: Save solution every N steps
        initial_condition: Per-DOF initial values (None = zero)
    """
    t_start: float = 0.0
    t_end: float = 1.0
    dt: float = 0.01
    scheme: str = "backward_euler"  # "backward_euler", "bdf2", "crank_nicolson"
    save_interval: int = 1
    initial_condition: Optional[np.ndarray] = None


@dataclass
class NonlinearConfig:
    """
    Newton iteration settings for nonlinear problems.

    None (default) means linear solve — existing behaviour is unchanged.

    Attributes:
        max_iter: Maximum Newton iterations
        abs_tol: Absolute residual tolerance
        rel_tol: Relative residual tolerance
        line_search: Enable line search in Newton iteration
    """
    max_iter: int = 20
    abs_tol: float = 1e-10
    rel_tol: float = 1e-8
    line_search: bool = True


@dataclass
class PhaseFieldConfig:
    """
    Configuration for phase field fracture model (AT2).

    Implements the Ambrosio-Tortorelli (AT2) regularization of brittle fracture.

    Attributes:
        G_c: Fracture toughness / critical energy release rate (J/m²)
        l_0: Regularization length scale (m) - controls crack width
        k_res: Residual stiffness to prevent numerical singularity
        stagger_tol: Convergence tolerance for staggered scheme
        stagger_max_iter: Maximum staggered iterations per load step
        n_load_steps: Number of quasi-static load steps
        damage_threshold: Threshold for crack path extraction (d > threshold)
    """
    G_c: float = 2.7e3       # Fracture toughness (J/m²) - typical for steel
    l_0: float = 0.015       # Regularization length (m)
    k_res: float = 1e-7      # Residual stiffness for numerical stability
    stagger_tol: float = 1e-4
    stagger_max_iter: int = 100
    n_load_steps: int = 50
    damage_threshold: float = 0.9  # d > 0.9 considered as crack

    def __post_init__(self):
        """Validate phase field parameters."""
        if self.G_c <= 0:
            raise ValueError("Fracture toughness G_c must be positive")
        if self.l_0 <= 0:
            raise ValueError("Regularization length l_0 must be positive")
        if not 0 < self.k_res < 1:
            raise ValueError("Residual stiffness k_res must be in (0, 1)")
        if not 0 < self.damage_threshold < 1:
            raise ValueError("Damage threshold must be in (0, 1)")


@dataclass
class PhysicsConfig:
    """
    Physics configuration for FEM analysis.

    Attributes:
        physics_type: Type of physics
        material: Material properties
        boundary_conditions: List of boundary conditions
        body_force: Body force vector (e.g., gravity)
        heat_source: Volumetric heat source (W/m^3)
        transient: Time integration settings (None = steady-state)
        nonlinear: Newton iteration settings (None = linear problem)
        phase_field: Phase field fracture settings (None = no fracture)
    """
    physics_type: PhysicsType
    material: MaterialProperties = field(default_factory=MaterialProperties)
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    body_force: Optional[np.ndarray] = None
    heat_source: float = 0.0
    transient: Optional[TransientConfig] = None
    nonlinear: Optional[NonlinearConfig] = None
    phase_field: Optional[PhaseFieldConfig] = None


@dataclass
class SolverResult:
    """
    Result of a solver execution.

    Attributes:
        success: Whether the solve completed successfully
        solution_data: Dictionary of solution field data
        output_files: List of output file paths
        metrics: Dictionary of computed metrics
        error_message: Error message if solve failed
        solve_time: Time taken for solve (seconds)
    """
    success: bool
    solution_data: Dict[str, np.ndarray] = field(default_factory=dict)
    output_files: List[Path] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    solve_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "solution_fields": list(self.solution_data.keys()),
            "output_files": [str(p) for p in self.output_files],
            "metrics": self.metrics,
            "error_message": self.error_message,
            "solve_time": self.solve_time,
        }


class SolverInterface(ABC):
    """
    Abstract base class for FEM solvers.

    Provides a unified interface for:
    - Setting up solver with mesh and physics configuration
    - Executing the solve
    - Extracting solution fields
    """

    def __init__(self):
        """Initialize the solver."""
        self._mesh_manager: Optional[MeshManager] = None
        self._physics: Optional[PhysicsConfig] = None
        self._is_setup: bool = False

    @abstractmethod
    def setup(
        self,
        mesh_manager: MeshManager,
        physics: PhysicsConfig
    ) -> None:
        """
        Set up the solver with mesh and physics configuration.

        Args:
            mesh_manager: Mesh manager instance
            physics: Physics configuration
        """
        self._mesh_manager = mesh_manager
        self._physics = physics

    @abstractmethod
    def solve(self, output_dir: Union[str, Path]) -> SolverResult:
        """
        Execute the solver.

        Args:
            output_dir: Directory to store output files

        Returns:
            SolverResult: Result of the solve
        """
        pass

    @abstractmethod
    def get_solution_field(self, field_name: str) -> np.ndarray:
        """
        Extract a solution field.

        Args:
            field_name: Name of the field to extract

        Returns:
            np.ndarray: Solution field data
        """
        pass

    @property
    def is_setup(self) -> bool:
        """Check if solver is set up."""
        return self._is_setup

    @property
    def mesh_manager(self) -> Optional[MeshManager]:
        """Get the mesh manager."""
        return self._mesh_manager

    @property
    def physics(self) -> Optional[PhysicsConfig]:
        """Get the physics configuration."""
        return self._physics

    def get_available_fields(self) -> List[str]:
        """
        Get list of available solution fields.

        Returns:
            List[str]: Names of available fields
        """
        return []
