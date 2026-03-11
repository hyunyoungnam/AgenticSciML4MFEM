"""
Solution data models for evolutionary tree search.

Defines the core data structures for representing solutions (model variants),
their genomes (mutation parameters), and metrics (quality scores).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
import json


class SolutionStatus(Enum):
    """Status of a solution in the evolutionary pipeline."""
    PROPOSED = "proposed"           # Initial proposal, not yet validated
    VALIDATED = "validated"         # Passed pre-flight checks
    EXECUTING = "executing"         # Solver is running
    EXECUTED = "executed"           # Solver finished (may have errors)
    CONVERGED = "converged"         # Solver converged successfully
    FAILED = "failed"               # Failed at some stage
    REJECTED = "rejected"           # Rejected by Critic during debate


@dataclass
class SolutionGenome:
    """
    Genome representing the mutation parameters for a solution.

    Encodes all the parameters that define how the base model
    was modified to create this solution variant.

    Attributes:
        delta_R: Hole radius change (for morphing)
        material_changes: Dict of material property modifications
        boundary_condition_changes: Dict of BC modifications
        mesh_refinement: Mesh refinement parameters
        step_changes: Solver step modifications
        custom_params: Any additional custom parameters
    """
    delta_R: float = 0.0
    material_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    boundary_condition_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mesh_refinement: Dict[str, Any] = field(default_factory=dict)
    step_changes: Dict[str, Any] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary representation."""
        return {
            "delta_R": self.delta_R,
            "material_changes": self.material_changes,
            "boundary_condition_changes": self.boundary_condition_changes,
            "mesh_refinement": self.mesh_refinement,
            "step_changes": self.step_changes,
            "custom_params": self.custom_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolutionGenome":
        """Create genome from dictionary representation."""
        return cls(
            delta_R=data.get("delta_R", 0.0),
            material_changes=data.get("material_changes", {}),
            boundary_condition_changes=data.get("boundary_condition_changes", {}),
            mesh_refinement=data.get("mesh_refinement", {}),
            step_changes=data.get("step_changes", {}),
            custom_params=data.get("custom_params", {}),
        )

    def get_mutation_summary(self) -> str:
        """Get a human-readable summary of the mutations."""
        parts = []

        if self.delta_R != 0.0:
            parts.append(f"delta_R={self.delta_R:+.3f}")

        if self.material_changes:
            mat_names = list(self.material_changes.keys())
            parts.append(f"materials={mat_names}")

        if self.boundary_condition_changes:
            bc_names = list(self.boundary_condition_changes.keys())
            parts.append(f"BCs={bc_names}")

        if self.mesh_refinement:
            parts.append(f"mesh_refined")

        if self.step_changes:
            parts.append(f"steps_modified")

        return ", ".join(parts) if parts else "no changes"

    def distance(self, other: "SolutionGenome") -> float:
        """
        Compute distance between two genomes.

        Used for diversity calculations in evolution.

        Args:
            other: Another genome to compare

        Returns:
            Distance metric (0 = identical)
        """
        dist = 0.0

        # Delta R difference (normalized)
        dist += abs(self.delta_R - other.delta_R) / 2.0

        # Material changes difference
        all_mats = set(self.material_changes.keys()) | set(other.material_changes.keys())
        for mat in all_mats:
            if mat not in self.material_changes:
                dist += 0.5
            elif mat not in other.material_changes:
                dist += 0.5
            else:
                # Compare properties
                props1 = self.material_changes[mat]
                props2 = other.material_changes[mat]
                for key in set(props1.keys()) | set(props2.keys()):
                    v1 = props1.get(key, 0)
                    v2 = props2.get(key, 0)
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        max_val = max(abs(v1), abs(v2), 1)
                        dist += abs(v1 - v2) / max_val * 0.1

        return dist


@dataclass
class SolutionMetrics:
    """
    Quality metrics for a solution.

    Captures various quality indicators from pre-flight checks
    and post-simulation analysis.

    Attributes:
        jacobian_min: Minimum Jacobian ratio
        jacobian_avg: Average Jacobian ratio
        aspect_ratio_max: Maximum element aspect ratio
        aspect_ratio_avg: Average element aspect ratio
        convergence_iterations: Number of solver iterations
        convergence_residual: Final residual
        stress_max: Maximum stress value
        displacement_max: Maximum displacement
        energy_balance: Energy balance error
        preflight_score: Pre-flight validation score (0-1)
        solver_score: Solver convergence score (0-1)
        quality_score: Overall mesh quality score (0-1)
        custom_metrics: Additional custom metrics
    """
    jacobian_min: Optional[float] = None
    jacobian_avg: Optional[float] = None
    aspect_ratio_max: Optional[float] = None
    aspect_ratio_avg: Optional[float] = None
    convergence_iterations: Optional[int] = None
    convergence_residual: Optional[float] = None
    stress_max: Optional[float] = None
    displacement_max: Optional[float] = None
    energy_balance: Optional[float] = None
    preflight_score: float = 0.0
    solver_score: float = 0.0
    quality_score: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "jacobian_min": self.jacobian_min,
            "jacobian_avg": self.jacobian_avg,
            "aspect_ratio_max": self.aspect_ratio_max,
            "aspect_ratio_avg": self.aspect_ratio_avg,
            "convergence_iterations": self.convergence_iterations,
            "convergence_residual": self.convergence_residual,
            "stress_max": self.stress_max,
            "displacement_max": self.displacement_max,
            "energy_balance": self.energy_balance,
            "preflight_score": self.preflight_score,
            "solver_score": self.solver_score,
            "quality_score": self.quality_score,
            "custom_metrics": self.custom_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolutionMetrics":
        """Create metrics from dictionary representation."""
        return cls(
            jacobian_min=data.get("jacobian_min"),
            jacobian_avg=data.get("jacobian_avg"),
            aspect_ratio_max=data.get("aspect_ratio_max"),
            aspect_ratio_avg=data.get("aspect_ratio_avg"),
            convergence_iterations=data.get("convergence_iterations"),
            convergence_residual=data.get("convergence_residual"),
            stress_max=data.get("stress_max"),
            displacement_max=data.get("displacement_max"),
            energy_balance=data.get("energy_balance"),
            preflight_score=data.get("preflight_score", 0.0),
            solver_score=data.get("solver_score", 0.0),
            quality_score=data.get("quality_score", 0.0),
            custom_metrics=data.get("custom_metrics", {}),
        )

    def compute_overall_score(self) -> float:
        """
        Compute an overall quality score.

        Returns:
            Score between 0 and 1 (higher is better)
        """
        scores = []

        # Preflight score
        if self.preflight_score > 0:
            scores.append(self.preflight_score)

        # Solver score
        if self.solver_score > 0:
            scores.append(self.solver_score)

        # Quality score
        if self.quality_score > 0:
            scores.append(self.quality_score)

        # Jacobian-based score
        if self.jacobian_min is not None:
            # Good Jacobian > 0.1, bad < 0.01
            jac_score = min(1.0, max(0.0, (self.jacobian_min - 0.01) / 0.09))
            scores.append(jac_score)

        # Aspect ratio score
        if self.aspect_ratio_max is not None:
            # Good AR < 10, bad > 100
            ar_score = min(1.0, max(0.0, 1.0 - (self.aspect_ratio_max - 1) / 99))
            scores.append(ar_score)

        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class DebateRound:
    """
    Record of a single round in the Proposer-Critic debate.

    Attributes:
        round_number: Which round (1-4)
        proposer_message: What the Proposer said
        critic_message: What the Critic said
        proposer_reasoning: Proposer's reasoning
        critic_objections: Critic's objections
        timestamp: When this round occurred
    """
    round_number: int
    proposer_message: str
    critic_message: str
    proposer_reasoning: Optional[str] = None
    critic_objections: Optional[List[str]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert debate round to dictionary representation."""
        return {
            "round_number": self.round_number,
            "proposer_message": self.proposer_message,
            "critic_message": self.critic_message,
            "proposer_reasoning": self.proposer_reasoning,
            "critic_objections": self.critic_objections,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Solution:
    """
    A solution in the evolutionary tree.

    Represents a specific model variant with its genome (parameters),
    status, metrics, and debate history.

    Attributes:
        id: Unique solution identifier
        parent_id: ID of the parent solution (None for root)
        generation: Generation number (0 for initial)
        genome: Mutation parameters defining this solution
        status: Current status in the pipeline
        metrics: Quality metrics
        inp_path: Path to the generated .inp file
        debate_rounds: History of Proposer-Critic debate
        error_messages: Any error messages encountered
        created_at: When the solution was created
        updated_at: When the solution was last updated
        metadata: Additional metadata
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    generation: int = 0
    genome: SolutionGenome = field(default_factory=SolutionGenome)
    status: SolutionStatus = SolutionStatus.PROPOSED
    metrics: SolutionMetrics = field(default_factory=SolutionMetrics)
    inp_path: Optional[str] = None
    debate_rounds: List[DebateRound] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary representation."""
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "genome": self.genome.to_dict(),
            "status": self.status.value,
            "metrics": self.metrics.to_dict(),
            "inp_path": self.inp_path,
            "debate_rounds": [r.to_dict() for r in self.debate_rounds],
            "error_messages": self.error_messages,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Solution":
        """Create solution from dictionary representation."""
        solution = cls(
            id=data["id"],
            parent_id=data.get("parent_id"),
            generation=data.get("generation", 0),
            genome=SolutionGenome.from_dict(data.get("genome", {})),
            status=SolutionStatus(data.get("status", "proposed")),
            metrics=SolutionMetrics.from_dict(data.get("metrics", {})),
            inp_path=data.get("inp_path"),
            error_messages=data.get("error_messages", []),
            metadata=data.get("metadata", {}),
        )

        if data.get("created_at"):
            solution.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            solution.updated_at = datetime.fromisoformat(data["updated_at"])

        return solution

    def update_status(self, new_status: SolutionStatus) -> None:
        """Update the solution status and timestamp."""
        self.status = new_status
        self.updated_at = datetime.now()

    def add_error(self, error: str) -> None:
        """Add an error message to the solution."""
        self.error_messages.append(error)
        self.updated_at = datetime.now()

    def add_debate_round(self, round: DebateRound) -> None:
        """Add a debate round to the history."""
        self.debate_rounds.append(round)
        self.updated_at = datetime.now()

    def is_successful(self) -> bool:
        """Check if the solution converged successfully."""
        return self.status == SolutionStatus.CONVERGED

    def is_terminal(self) -> bool:
        """Check if the solution has reached a terminal state."""
        return self.status in (
            SolutionStatus.CONVERGED,
            SolutionStatus.FAILED,
            SolutionStatus.REJECTED,
        )

    def get_lineage(self) -> List[str]:
        """
        Get the lineage of solution IDs from root to this solution.

        Note: This only returns the IDs, actual lookup needs the tree.
        """
        lineage = [self.id]
        # Parent tracking would need to be done by the tree
        return lineage

    def to_json(self, indent: int = 2) -> str:
        """Convert solution to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Solution":
        """Create solution from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return (
            f"Solution(id={self.id[:8]}..., gen={self.generation}, "
            f"status={self.status.value}, genome={self.genome.get_mutation_summary()})"
        )
