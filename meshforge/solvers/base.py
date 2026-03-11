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


class BoundaryConditionType(Enum):
    """Enumeration of boundary condition types."""
    # Mechanical
    DISPLACEMENT = auto()  # Fixed displacement
    TRACTION = auto()      # Applied force/traction
    SYMMETRY = auto()      # Symmetry condition
    # Thermal
    TEMPERATURE = auto()   # Fixed temperature (Dirichlet)
    HEAT_FLUX = auto()     # Applied heat flux (Neumann)
    CONVECTION = auto()    # Convective boundary


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
    """
    E: float = 200e9        # Default: Steel Young's modulus
    nu: float = 0.3         # Default: Steel Poisson's ratio
    density: float = 7850.0 # Default: Steel density
    k: float = 50.0         # Default: Steel thermal conductivity
    cp: float = 500.0       # Default: Steel specific heat

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
        value: BC value (scalar, vector, or None for symmetry)
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
class PhysicsConfig:
    """
    Physics configuration for FEM analysis.

    Attributes:
        physics_type: Type of physics (elasticity, heat transfer, etc.)
        material: Material properties
        boundary_conditions: List of boundary conditions
        body_force: Body force vector (e.g., gravity)
        heat_source: Volumetric heat source (W/m^3)
    """
    physics_type: PhysicsType
    material: MaterialProperties = field(default_factory=MaterialProperties)
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    body_force: Optional[np.ndarray] = None
    heat_source: float = 0.0


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
