"""
Evaluation pipeline for FEA solutions.

Coordinates pre-flight checks, solver execution, and post-simulation
metric extraction for solution evaluation.
"""

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from inpforge.evaluation.preflight import PreflightChecker, PreflightResult, PreflightStatus
from inpforge.evaluation.metrics import MetricsCalculator, MeshQualityMetrics, calculate_solution_score


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
    Full evaluation pipeline for FEA solutions.

    Coordinates:
    1. Pre-flight validation
    2. Solver execution (optional)
    3. Post-simulation analysis
    4. Score calculation
    """

    def __init__(
        self,
        preflight_checker: Optional[PreflightChecker] = None,
        metrics_calculator: Optional[MetricsCalculator] = None,
        abaqus_cmd: str = "abaqus",
        run_solver: bool = False,
        solver_timeout: int = 3600,
    ):
        """
        Initialize the evaluation pipeline.

        Args:
            preflight_checker: PreflightChecker instance
            metrics_calculator: MetricsCalculator instance
            abaqus_cmd: Abaqus command
            run_solver: Whether to run the solver
            solver_timeout: Solver timeout in seconds
        """
        self.preflight_checker = preflight_checker or PreflightChecker()
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.abaqus_cmd = abaqus_cmd
        self.run_solver = run_solver
        self.solver_timeout = solver_timeout

    def evaluate(
        self,
        solution_id: str,
        inp_path: str,
        skip_solver: bool = True,
    ) -> EvaluationResult:
        """
        Run the full evaluation pipeline.

        Args:
            solution_id: Solution identifier
            inp_path: Path to the .inp file
            skip_solver: Whether to skip solver execution

        Returns:
            EvaluationResult
        """
        result = EvaluationResult(solution_id=solution_id)

        # Stage 1: Pre-flight validation
        try:
            result.stage = EvaluationStage.PREFLIGHT
            result.preflight_result = self.preflight_checker.check_file(inp_path)

            result.errors.extend(result.preflight_result.errors)
            result.warnings.extend(result.preflight_result.warnings)

            if result.preflight_result.status == PreflightStatus.FAILED:
                result.stage = EvaluationStage.FAILED
                result.success = False
                result.completed_at = datetime.now()
                return result

        except Exception as e:
            result.errors.append(f"Preflight failed: {str(e)}")
            result.stage = EvaluationStage.FAILED
            result.completed_at = datetime.now()
            return result

        # Calculate mesh metrics
        try:
            from manager import AbaqusManager
            manager = AbaqusManager(inp_path)
            result.mesh_metrics = self.metrics_calculator.calculate_from_manager(manager)
        except Exception as e:
            result.warnings.append(f"Mesh metrics calculation failed: {str(e)}")

        # Stage 2: Solver execution (optional)
        if self.run_solver and not skip_solver:
            try:
                result.stage = EvaluationStage.SOLVER
                solver_result = self._run_solver(inp_path)
                result.solver_completed = solver_result["completed"]
                result.solver_output = solver_result.get("output", "")

                if not solver_result["completed"]:
                    result.errors.append(solver_result.get("error", "Solver failed"))
                    result.stage = EvaluationStage.FAILED
                    result.success = False
                    result.completed_at = datetime.now()
                    return result

                # Extract convergence metrics
                result.convergence_metrics = self.metrics_calculator.calculate_convergence_metrics(
                    result.solver_output
                )

            except Exception as e:
                result.errors.append(f"Solver execution failed: {str(e)}")
                result.stage = EvaluationStage.FAILED
                result.completed_at = datetime.now()
                return result
        else:
            result.solver_completed = True  # Assume success if not running

        # Stage 3: Post-analysis and scoring
        result.stage = EvaluationStage.POST_ANALYSIS

        # Calculate overall score
        preflight_score = result.preflight_result.score if result.preflight_result else 0.5

        if result.mesh_metrics:
            result.overall_score = calculate_solution_score(
                result.mesh_metrics,
                preflight_score,
                result.convergence_metrics if self.run_solver else None,
            )
        else:
            result.overall_score = preflight_score * 0.5

        # Mark complete
        result.stage = EvaluationStage.COMPLETE
        result.success = (
            result.preflight_result.is_valid if result.preflight_result else False
        ) and result.solver_completed
        result.completed_at = datetime.now()

        return result

    def _run_solver(self, inp_path: str) -> Dict[str, Any]:
        """
        Run the Abaqus solver.

        Args:
            inp_path: Path to .inp file

        Returns:
            Dict with solver result
        """
        result = {
            "completed": False,
            "output": "",
            "error": None,
        }

        try:
            inp_path = Path(inp_path)
            job_name = inp_path.stem
            work_dir = inp_path.parent

            # Build command
            cmd = [
                self.abaqus_cmd,
                "job=" + job_name,
                "input=" + str(inp_path),
                "interactive",
            ]

            # Run solver
            process = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.solver_timeout,
            )

            result["output"] = process.stdout + process.stderr

            # Check for success
            if process.returncode == 0:
                result["completed"] = True
            else:
                result["error"] = f"Solver returned code {process.returncode}"

            # Check for completion message in output
            if "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in result["output"]:
                result["completed"] = True

        except subprocess.TimeoutExpired:
            result["error"] = f"Solver timed out after {self.solver_timeout}s"

        except FileNotFoundError:
            result["error"] = f"Abaqus command not found: {self.abaqus_cmd}"

        except Exception as e:
            result["error"] = str(e)

        return result

    def quick_evaluate(
        self,
        solution_id: str,
        inp_path: str,
    ) -> EvaluationResult:
        """
        Quick evaluation without solver execution.

        Args:
            solution_id: Solution identifier
            inp_path: Path to .inp file

        Returns:
            EvaluationResult
        """
        return self.evaluate(solution_id, inp_path, skip_solver=True)

    def evaluate_manager(
        self,
        solution_id: str,
        manager,
    ) -> EvaluationResult:
        """
        Evaluate an AbaqusManager instance directly.

        Args:
            solution_id: Solution identifier
            manager: AbaqusManager instance

        Returns:
            EvaluationResult
        """
        result = EvaluationResult(solution_id=solution_id)

        # Pre-flight
        result.stage = EvaluationStage.PREFLIGHT
        result.preflight_result = self.preflight_checker.check_manager(manager)
        result.errors.extend(result.preflight_result.errors)
        result.warnings.extend(result.preflight_result.warnings)

        if result.preflight_result.status == PreflightStatus.FAILED:
            result.stage = EvaluationStage.FAILED
            result.success = False
            result.completed_at = datetime.now()
            return result

        # Mesh metrics
        result.mesh_metrics = self.metrics_calculator.calculate_from_manager(manager)

        # Score
        preflight_score = result.preflight_result.score
        if result.mesh_metrics:
            result.overall_score = calculate_solution_score(
                result.mesh_metrics,
                preflight_score,
                None,
            )
        else:
            result.overall_score = preflight_score * 0.5

        result.stage = EvaluationStage.COMPLETE
        result.success = result.preflight_result.is_valid
        result.solver_completed = True
        result.completed_at = datetime.now()

        return result
