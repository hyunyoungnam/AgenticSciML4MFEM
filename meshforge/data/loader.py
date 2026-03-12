"""
Data loaders for FEM simulation results.

Provides utilities for loading FEM results from various sources
(MFEM output files, etc.) into FEMSample objects.
"""

import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .dataset import FEMSample


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load_sample(
        self,
        mesh_path: Path,
        solution_path: Optional[Path] = None,
        parameters: Optional[Dict[str, float]] = None
    ) -> FEMSample:
        """
        Load a single FEM sample.

        Args:
            mesh_path: Path to mesh file
            solution_path: Path to solution file(s)
            parameters: Input parameters for this sample

        Returns:
            FEMSample object
        """
        pass

    @abstractmethod
    def load_batch(
        self,
        directory: Path,
        pattern: str = "*.mesh"
    ) -> List[FEMSample]:
        """
        Load multiple samples from a directory.

        Args:
            directory: Directory containing samples
            pattern: Glob pattern for finding files

        Returns:
            List of FEMSample objects
        """
        pass


class MFEMDataLoader(DatasetLoader):
    """
    Loader for MFEM simulation results.

    Loads mesh and solution data from MFEM output files.
    """

    def __init__(self):
        """Initialize loader."""
        self._mfem = None

    def _get_mfem(self):
        """Lazy import of mfem module."""
        if self._mfem is None:
            try:
                import mfem.ser as mfem
                self._mfem = mfem
            except ImportError:
                raise ImportError(
                    "PyMFEM is required for MFEM data loading. "
                    "Install with: pip install mfem"
                )
        return self._mfem

    def load_sample(
        self,
        mesh_path: Path,
        solution_path: Optional[Path] = None,
        parameters: Optional[Dict[str, float]] = None
    ) -> FEMSample:
        """
        Load a sample from MFEM files.

        Args:
            mesh_path: Path to .mesh file
            solution_path: Path to .gf solution file (optional)
            parameters: Input parameters for this sample

        Returns:
            FEMSample object
        """
        mfem = self._get_mfem()
        mesh_path = Path(mesh_path)

        # Load mesh
        mesh = mfem.Mesh(str(mesh_path))
        dim = mesh.SpaceDimension()

        # Extract coordinates
        num_vertices = mesh.GetNV()
        coordinates = np.zeros((num_vertices, dim), dtype=np.float64)
        for i in range(num_vertices):
            vertex = mesh.GetVertex(i)
            for d in range(dim):
                coordinates[i, d] = vertex[d]

        # Initialize fields
        displacement = None
        stress = None
        temperature = None
        von_mises = None

        # Load solution if provided
        if solution_path and solution_path.exists():
            displacement, stress, von_mises = self._load_solution(
                mesh, solution_path, dim
            )

        # Generate sample ID
        sample_id = f"sample_{mesh_path.stem}_{uuid.uuid4().hex[:8]}"

        return FEMSample(
            sample_id=sample_id,
            parameters=parameters or {},
            coordinates=coordinates,
            displacement=displacement,
            stress=stress,
            von_mises=von_mises,
            mesh_file=mesh_path,
            metadata={
                "dimension": dim,
                "n_elements": mesh.GetNE(),
            },
            is_valid=True,
        )

    def _load_solution(
        self,
        mesh,
        solution_path: Path,
        dim: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load solution from GridFunction file."""
        mfem = self._get_mfem()

        try:
            # Read the GridFunction
            with open(solution_path, "r") as f:
                gf = mfem.GridFunction(mesh, f)

            num_vertices = mesh.GetNV()
            vdim = gf.VectorDim()

            # Extract field values
            if vdim == 1:
                # Scalar field (temperature, etc.)
                temperature = np.zeros(num_vertices, dtype=np.float64)
                for i in range(num_vertices):
                    temperature[i] = gf[i]
                return None, None, None  # Return temperature separately

            else:
                # Vector field (displacement)
                fespace = gf.FESpace()
                displacement = np.zeros((num_vertices, dim), dtype=np.float64)

                for i in range(num_vertices):
                    for d in range(dim):
                        dof = fespace.DofToVDof(i, d)
                        displacement[i, d] = gf[dof]

                return displacement, None, None

        except Exception:
            return None, None, None

    def load_batch(
        self,
        directory: Path,
        pattern: str = "*.mesh"
    ) -> List[FEMSample]:
        """
        Load multiple samples from a directory.

        Expects directory structure:
        directory/
            sample_001/
                mesh.mesh
                solution.gf
                params.json
            sample_002/
                ...

        Args:
            directory: Directory containing samples
            pattern: Glob pattern for finding mesh files

        Returns:
            List of FEMSample objects
        """
        import json

        directory = Path(directory)
        samples = []

        # Find all mesh files
        mesh_files = sorted(directory.glob(f"**/{pattern}"))

        for mesh_path in mesh_files:
            sample_dir = mesh_path.parent

            # Look for solution file
            solution_path = None
            for ext in [".gf", "_solution.gf", "_displacement.gf"]:
                candidate = sample_dir / f"{mesh_path.stem}{ext}"
                if candidate.exists():
                    solution_path = candidate
                    break

            # Look for parameters file
            params = {}
            params_path = sample_dir / "params.json"
            if params_path.exists():
                with open(params_path, "r") as f:
                    params = json.load(f)

            try:
                sample = self.load_sample(mesh_path, solution_path, params)
                samples.append(sample)
            except Exception as e:
                print(f"Warning: Failed to load {mesh_path}: {e}")

        return samples


class SimulationResultLoader:
    """
    Loader that creates samples from simulation results.

    Used to convert MFEMSolver output to FEMSample format.
    """

    def __init__(self):
        pass

    def from_solver_result(
        self,
        solver_result: Any,  # SolverResult from mfem_solver
        mesh_manager: Any,   # MFEMManager
        parameters: Dict[str, float],
        sample_id: Optional[str] = None
    ) -> FEMSample:
        """
        Create FEMSample from solver result.

        Args:
            solver_result: Result from MFEMSolver.solve()
            mesh_manager: MFEMManager instance
            parameters: Input parameters
            sample_id: Optional sample ID

        Returns:
            FEMSample object
        """
        if sample_id is None:
            sample_id = f"sim_{uuid.uuid4().hex[:8]}"

        # Extract coordinates from mesh manager
        coordinates = mesh_manager.get_nodes()

        # Extract solution fields
        solution_data = solver_result.solution_data

        return FEMSample(
            sample_id=sample_id,
            parameters=parameters,
            coordinates=coordinates,
            displacement=solution_data.get("displacement"),
            stress=solution_data.get("stress"),
            temperature=solution_data.get("temperature"),
            von_mises=solution_data.get("von_mises"),
            mesh_file=mesh_manager.file_path if hasattr(mesh_manager, "file_path") else None,
            metadata={
                "solver_metrics": solver_result.metrics,
                "solve_time": solver_result.solve_time,
            },
            is_valid=solver_result.success,
        )
