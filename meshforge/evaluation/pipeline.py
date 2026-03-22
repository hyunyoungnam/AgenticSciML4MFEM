"""
Evaluation pipeline for FEA solutions.

Coordinates mesh validation, solver execution, and post-simulation
metric extraction for solution evaluation using MFEM.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from meshforge.evaluation.metrics import MetricsCalculator, MeshQualityMetrics, calculate_solution_score
from meshforge.mesh.base import MeshManager
from meshforge.solvers.base import SolverInterface, PhysicsConfig, SolverResult


class PreflightStatus(Enum):
    """Status of pre-flight mesh validation."""
    PASSED = "passed"
    WARNINGS = "warnings"
    FAILED = "failed"


@dataclass
class PreflightResult:
    """Result of pre-flight mesh validation."""
    status: PreflightStatus = PreflightStatus.PASSED
    score: float = 1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "score": self.score,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class EvaluationStage(Enum):
    """Stages of the evaluation pipeline."""
    PREFLIGHT = "preflight"
    SOLVER = "solver"
    POST_ANALYSIS = "post_analysis"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class EvaluationResult:
    """Result of the full evaluation pipeline."""
    solution_id: str
    stage: EvaluationStage = EvaluationStage.PREFLIGHT
    success: bool = False
    preflight_result: Optional[PreflightResult] = None
    mesh_metrics: Optional[MeshQualityMetrics] = None
    solver_completed: bool = False
    solver_output: str = ""
    convergence_metrics: Dict[str, Any] = field(default_factory=dict)
    overall_score: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solution_id": self.solution_id,
            "stage": self.stage.value,
            "success": self.success,
            "preflight_result": self.preflight_result.to_dict() if self.preflight_result else None,
            "mesh_metrics": self.mesh_metrics.to_dict() if self.mesh_metrics else None,
            "solver_completed": self.solver_completed,
            "convergence_metrics": self.convergence_metrics,
            "overall_score": self.overall_score,
            "errors": self.errors,
            "warnings": self.warnings,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    def get_summary(self) -> str:
        """Get a summary of the evaluation."""
        lines = [
            f"Solution: {self.solution_id}",
            f"Stage: {self.stage.value}",
            f"Success: {self.success}",
            f"Overall Score: {self.overall_score:.3f}",
        ]

        if self.preflight_result:
            lines.append(f"Preflight: {self.preflight_result.status.value}")

        if self.solver_completed:
            lines.append(f"Solver: completed")
            if self.convergence_metrics.get("converged"):
                lines.append("  Converged: Yes")
            else:
                lines.append("  Converged: No")

        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")

        return "\n".join(lines)


class EvaluationPipeline:
    """
    Full evaluation pipeline for FEA solutions using MFEM.

    Coordinates:
    1. Mesh validation
    2. Solver execution (optional, via MFEM)
    3. Post-simulation analysis
    4. Score calculation
    """

    def __init__(
        self,
        metrics_calculator: Optional[MetricsCalculator] = None,
        run_solver: bool = False,
        solver_timeout: int = 3600,
        solver: Optional[SolverInterface] = None,
        physics: Optional[PhysicsConfig] = None,
    ):
        """
        Initialize the evaluation pipeline.

        Args:
            metrics_calculator: MetricsCalculator instance
            run_solver: Whether to run the solver
            solver_timeout: Solver timeout in seconds
            solver: Optional SolverInterface instance (MFEMSolver)
            physics: Optional PhysicsConfig for solver setup
        """
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self._run_solver = run_solver
        self.solver_timeout = solver_timeout
        self.solver = solver
        self.physics = physics

    def evaluate(
        self,
        solution_id: str,
        mesh_manager: MeshManager,
        output_dir: Optional[str] = None,
        skip_solver: bool = False,
    ) -> EvaluationResult:
        """
        Run the full evaluation pipeline.

        Args:
            solution_id: Solution identifier
            mesh_manager: MeshManager instance (MFEMManager)
            output_dir: Directory for solver output files
            skip_solver: Whether to skip solver execution

        Returns:
            EvaluationResult
        """
        result = EvaluationResult(solution_id=solution_id)

        # Stage 1: Basic mesh validation
        result.stage = EvaluationStage.PREFLIGHT

        try:
            # Validate mesh data is accessible
            nodes = mesh_manager.get_nodes()
            elements = mesh_manager.get_elements()

            if len(nodes) == 0:
                result.errors.append("Mesh has no nodes")
                result.stage = EvaluationStage.FAILED
                result.success = False
                result.completed_at = datetime.now()
                return result

            if len(elements) == 0:
                result.errors.append("Mesh has no elements")
                result.stage = EvaluationStage.FAILED
                result.success = False
                result.completed_at = datetime.now()
                return result

            # Create a preflight result for mesh validation
            result.preflight_result = PreflightResult(
                status=PreflightStatus.PASSED,
                score=1.0,
                errors=[],
                warnings=[],
                metadata={
                    "num_nodes": mesh_manager.num_nodes,
                    "num_elements": mesh_manager.num_elements,
                    "dimension": mesh_manager.dimension,
                }
            )

        except Exception as e:
            result.errors.append(f"Mesh validation failed: {str(e)}")
            result.stage = EvaluationStage.FAILED
            result.completed_at = datetime.now()
            return result

        # Calculate mesh metrics
        try:
            result.mesh_metrics = self.metrics_calculator.calculate_from_arrays(
                nodes, elements
            )
        except Exception as e:
            result.warnings.append(f"Mesh metrics calculation failed: {str(e)}")

        # Stage 2: Solver execution (if configured and not skipped)
        if self.solver is not None and self._run_solver and not skip_solver:
            try:
                result.stage = EvaluationStage.SOLVER

                # Setup solver with mesh and physics
                if self.physics is not None:
                    self.solver.setup(mesh_manager, self.physics)

                # Determine output directory
                if output_dir is None:
                    output_dir = Path(".") / "output" / solution_id
                else:
                    output_dir = Path(output_dir)

                # Run solver
                solver_result = self.solver.solve(output_dir)

                result.solver_completed = solver_result.success
                result.solver_output = str(solver_result.to_dict())

                if not solver_result.success:
                    result.errors.append(
                        solver_result.error_message or "Solver failed"
                    )
                    result.stage = EvaluationStage.FAILED
                    result.success = False
                    result.completed_at = datetime.now()
                    return result

                # Store solver metrics
                result.convergence_metrics = solver_result.metrics

            except Exception as e:
                result.errors.append(f"Solver execution failed: {str(e)}")
                result.stage = EvaluationStage.FAILED
                result.completed_at = datetime.now()
                return result
        else:
            result.solver_completed = True  # Assume success if not running

        # Stage 3: Post-analysis and scoring
        result.stage = EvaluationStage.POST_ANALYSIS

        preflight_score = (
            result.preflight_result.score if result.preflight_result else 0.5
        )

        if result.mesh_metrics:
            result.overall_score = calculate_solution_score(
                result.mesh_metrics,
                preflight_score,
                result.convergence_metrics if self._run_solver else None,
            )
        else:
            result.overall_score = preflight_score * 0.5

        # Mark complete
        result.stage = EvaluationStage.COMPLETE
        result.success = result.solver_completed
        result.completed_at = datetime.now()

        return result

    def quick_evaluate(
        self,
        solution_id: str,
        mesh_manager: MeshManager,
    ) -> EvaluationResult:
        """
        Quick evaluation without solver execution.

        Args:
            solution_id: Solution identifier
            mesh_manager: MeshManager instance

        Returns:
            EvaluationResult
        """
        return self.evaluate(solution_id, mesh_manager, skip_solver=True)

    def run_solver(
        self,
        mesh_manager: MeshManager,
        output_dir: str,
    ) -> SolverResult:
        """
        Run solver using the SolverInterface abstraction.

        Args:
            mesh_manager: MeshManager instance
            output_dir: Directory for output files

        Returns:
            SolverResult from the solver
        """
        if self.solver is None:
            return SolverResult(
                success=False,
                error_message="No solver configured. Set self.solver to a SolverInterface instance."
            )

        if self.physics is not None:
            self.solver.setup(mesh_manager, self.physics)

        return self.solver.solve(output_dir)
