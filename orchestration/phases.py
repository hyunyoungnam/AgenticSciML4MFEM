"""
Phase controllers for the AgenticSciML workflow.

Implements the four main phases:
1. Analysis & Evaluation Contract
2. Knowledge Funnel
3. Proposer-Critic Debate
4. Engineer + Debugger Loop
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import AgentContext
from agents.roles.evaluator import EvaluatorAgent, EvaluationCriteria
from agents.roles.proposer import ProposerAgent, MutationProposal
from agents.roles.critic import CriticAgent
from agents.roles.engineer import EngineerAgent, ImplementationResult
from agents.roles.debugger import DebuggerAgent, DebugAnalysis
from agents.roles.result_analyst import ResultAnalystAgent, ResultAnalysis
from agents.debate.controller import DebateController, DebateResult, DebateOutcome
from evolution.solution import Solution, SolutionStatus, SolutionGenome
from evaluation.pipeline import EvaluationPipeline, EvaluationResult


class PhaseStatus(Enum):
    """Status of a phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PhaseResult:
    """Result from a phase execution."""
    phase_name: str
    status: PhaseStatus
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "status": self.status.value,
            "data": self.data,
            "errors": self.errors,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Phase1AnalysisController:
    """
    Phase 1: Analysis & Evaluation Contract.

    1. Evaluator analyzes base .inp model
    2. Generates Guideline.md (mesh bounds, unit system, quality thresholds)
    3. Generates Evaluate.py (scoring script)
    4. Human approves evaluation criteria
    """

    def __init__(self, evaluator: EvaluatorAgent):
        self.evaluator = evaluator

    async def execute(
        self,
        context: AgentContext,
        output_dir: str,
        require_approval: bool = True,
    ) -> PhaseResult:
        """
        Execute Phase 1.

        Args:
            context: Agent context with model info
            output_dir: Directory for output files
            require_approval: Whether to require human approval

        Returns:
            PhaseResult with evaluation criteria
        """
        result = PhaseResult(
            phase_name="phase1_analysis",
            status=PhaseStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            # Run full evaluation setup
            criteria = await self.evaluator.full_evaluation_setup(
                context,
                output_dir=output_dir,
            )

            # Store results
            result.data["guidelines_path"] = str(Path(output_dir) / "Guideline.md")
            result.data["evaluate_path"] = str(Path(output_dir) / "Evaluate.py")
            result.data["analysis"] = criteria.analysis.raw_analysis if criteria.analysis else ""

            # Update context with evaluation criteria
            context.evaluation_criteria = {
                "guidelines": criteria.guidelines_md,
                "evaluate_script": criteria.evaluate_py,
            }

            if criteria.analysis:
                context.evaluation_criteria["analysis"] = criteria.analysis.to_dict() if hasattr(criteria.analysis, 'to_dict') else str(criteria.analysis)

            result.status = PhaseStatus.COMPLETED

        except Exception as e:
            result.status = PhaseStatus.FAILED
            result.errors.append(str(e))

        result.completed_at = datetime.now()
        return result


class Phase2KnowledgeController:
    """
    Phase 2: Knowledge Funnel.

    1. Retriever searches knowledge base for relevant FEA techniques
    2. Loads failure memory from previous runs
    3. Builds context for Proposer
    """

    def __init__(
        self,
        knowledge_base=None,
        failure_memory=None,
    ):
        self.knowledge_base = knowledge_base
        self.failure_memory = failure_memory

    async def execute(
        self,
        context: AgentContext,
        query: str = "",
        problem_type: str = "",
    ) -> PhaseResult:
        """
        Execute Phase 2.

        Args:
            context: Agent context
            query: Search query for knowledge base
            problem_type: Problem type for filtering

        Returns:
            PhaseResult with retrieved knowledge
        """
        result = PhaseResult(
            phase_name="phase2_knowledge",
            status=PhaseStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            # Search knowledge base
            if self.knowledge_base:
                if query:
                    entries = self.knowledge_base.search(query, top_k=10)
                elif problem_type:
                    entries = self.knowledge_base.search_by_problem_type(problem_type)
                else:
                    # Get general best practices
                    entries = self.knowledge_base.get_by_category("best_practices")

                context.knowledge_context = [e.to_dict() for e in entries]
                result.data["knowledge_entries"] = len(entries)
            else:
                result.data["knowledge_entries"] = 0

            # Load failure memory
            if self.failure_memory:
                failures = self.failure_memory.get_recent_failures(limit=20)
                context.failure_history = failures
                result.data["failure_entries"] = len(failures)
            else:
                result.data["failure_entries"] = 0

            result.status = PhaseStatus.COMPLETED

        except Exception as e:
            result.status = PhaseStatus.FAILED
            result.errors.append(str(e))

        result.completed_at = datetime.now()
        return result


class Phase3DebateController:
    """
    Phase 3: Proposer-Critic Debate.

    1. Round 1-2: Proposer presents reasoning, Critic challenges
    2. Round 3: Proposer synthesizes feedback
    3. Round 4: Final proposal, Critic votes APPROVE/REJECT
    """

    def __init__(
        self,
        proposer: ProposerAgent,
        critic: CriticAgent,
        num_rounds: int = 4,
    ):
        self.debate_controller = DebateController(
            proposer=proposer,
            critic=critic,
            num_rounds=num_rounds,
        )

    async def execute(
        self,
        context: AgentContext,
        solution: Solution,
        parent_mutations: str = "",
    ) -> PhaseResult:
        """
        Execute Phase 3.

        Args:
            context: Agent context
            solution: Current solution being proposed
            parent_mutations: Description of parent mutations

        Returns:
            PhaseResult with debate outcome
        """
        result = PhaseResult(
            phase_name="phase3_debate",
            status=PhaseStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            context.current_solution_id = solution.id

            # Run debate
            debate_result = await self.debate_controller.run_debate(
                context=context,
                generation=solution.generation,
                parent_mutations=parent_mutations,
            )

            # Store results
            result.data["outcome"] = debate_result.outcome.value
            result.data["total_rounds"] = debate_result.total_rounds
            result.data["consensus_score"] = debate_result.consensus_score

            if debate_result.final_proposal:
                result.data["proposal"] = debate_result.final_proposal.to_dict()

                # Update solution genome
                solution.genome = debate_result.final_proposal.to_genome()

            # Add debate rounds to solution
            for debate_round in debate_result.rounds:
                solution.add_debate_round(debate_round)

            # Update solution status based on outcome
            if debate_result.outcome == DebateOutcome.APPROVED:
                solution.update_status(SolutionStatus.VALIDATED)
                result.status = PhaseStatus.COMPLETED
            elif debate_result.outcome == DebateOutcome.REJECTED:
                solution.update_status(SolutionStatus.REJECTED)
                result.status = PhaseStatus.COMPLETED
            else:
                # Timeout - treat as soft rejection
                solution.update_status(SolutionStatus.REJECTED)
                result.status = PhaseStatus.COMPLETED

            result.data["solution_status"] = solution.status.value

        except Exception as e:
            result.status = PhaseStatus.FAILED
            result.errors.append(str(e))
            solution.add_error(str(e))

        result.completed_at = datetime.now()
        return result


class Phase4ExecutionController:
    """
    Phase 4: Engineer + Debugger Loop.

    1. Engineer implements approved mutations via manager.py + morphing.py
    2. Runs pre-flight validation via validator.py
    3. Executes Abaqus solver (optional)
    4. If error: Debugger diagnoses, Engineer retries (max 3 attempts)
    5. Result Analyst extracts metrics and success traits
    """

    def __init__(
        self,
        engineer: EngineerAgent,
        debugger: DebuggerAgent,
        result_analyst: ResultAnalystAgent,
        evaluation_pipeline: EvaluationPipeline,
        max_attempts: int = 3,
    ):
        self.engineer = engineer
        self.debugger = debugger
        self.result_analyst = result_analyst
        self.evaluation_pipeline = evaluation_pipeline
        self.max_attempts = max_attempts

    async def execute(
        self,
        context: AgentContext,
        solution: Solution,
        base_inp_path: str,
        output_dir: str,
        config_path: Optional[str] = None,
        run_solver: bool = False,
    ) -> PhaseResult:
        """
        Execute Phase 4.

        Args:
            context: Agent context
            solution: Solution to implement
            base_inp_path: Path to base .inp file
            output_dir: Directory for outputs
            config_path: Path to morphing config
            run_solver: Whether to run the Abaqus solver

        Returns:
            PhaseResult with execution results
        """
        result = PhaseResult(
            phase_name="phase4_execution",
            status=PhaseStatus.RUNNING,
            started_at=datetime.now(),
        )

        output_path = str(Path(output_dir) / f"{solution.id}.inp")
        attempt = 0
        implementation_result = None
        evaluation_result = None

        try:
            solution.update_status(SolutionStatus.EXECUTING)

            while attempt < self.max_attempts:
                attempt += 1
                result.data[f"attempt_{attempt}"] = {}

                # Step 1: Implement mutation
                proposal = MutationProposal(
                    delta_R=solution.genome.delta_R,
                    material_changes=solution.genome.material_changes,
                    bc_changes=solution.genome.boundary_condition_changes,
                )

                implementation_result = self.engineer.implement_mutation(
                    proposal=proposal,
                    base_inp_path=base_inp_path,
                    output_path=output_path,
                    config_path=config_path,
                )

                result.data[f"attempt_{attempt}"]["implementation"] = implementation_result.to_dict()

                if not implementation_result.success:
                    # Diagnose error
                    debug_analysis = await self.debugger.diagnose_error(
                        context=context,
                        error_type="implementation",
                        error_message=implementation_result.error_message or "Unknown error",
                        mutation=str(solution.genome.to_dict()),
                        delta_R=solution.genome.delta_R,
                    )

                    result.data[f"attempt_{attempt}"]["debug"] = debug_analysis.to_dict()

                    # Apply fix if suggested
                    if debug_analysis.suggested_delta_R is not None:
                        solution.genome.delta_R = debug_analysis.suggested_delta_R

                    continue

                # Step 2: Evaluate
                evaluation_result = self.evaluation_pipeline.evaluate(
                    solution_id=solution.id,
                    inp_path=output_path,
                    skip_solver=not run_solver,
                )

                result.data[f"attempt_{attempt}"]["evaluation"] = evaluation_result.to_dict()

                if evaluation_result.success:
                    # Success!
                    solution.inp_path = output_path
                    solution.metrics.preflight_score = evaluation_result.preflight_result.score if evaluation_result.preflight_result else 0.0
                    solution.metrics.quality_score = evaluation_result.overall_score

                    if evaluation_result.mesh_metrics:
                        solution.metrics.jacobian_min = evaluation_result.mesh_metrics.jacobian_min
                        solution.metrics.aspect_ratio_max = evaluation_result.mesh_metrics.aspect_ratio_max

                    solution.update_status(SolutionStatus.CONVERGED)
                    break

                else:
                    # Evaluation failed - diagnose
                    error_msg = "; ".join(evaluation_result.errors) if evaluation_result.errors else "Evaluation failed"
                    solution.add_error(error_msg)

                    debug_analysis = await self.debugger.diagnose_error(
                        context=context,
                        error_type="evaluation",
                        error_message=error_msg,
                        mutation=str(solution.genome.to_dict()),
                        delta_R=solution.genome.delta_R,
                    )

                    result.data[f"attempt_{attempt}"]["debug"] = debug_analysis.to_dict()

                    if debug_analysis.suggested_delta_R is not None:
                        solution.genome.delta_R = debug_analysis.suggested_delta_R

            # Final status
            result.data["total_attempts"] = attempt

            if solution.status != SolutionStatus.CONVERGED:
                solution.update_status(SolutionStatus.FAILED)
                result.status = PhaseStatus.FAILED
            else:
                # Run result analysis
                if evaluation_result:
                    analysis = await self.result_analyst.analyze_results(
                        context=context,
                        solution_id=solution.id,
                        delta_R=solution.genome.delta_R,
                        mutation_type=solution.genome.get_mutation_summary(),
                        mesh_metrics=str(evaluation_result.mesh_metrics.to_dict()) if evaluation_result.mesh_metrics else "",
                    )
                    result.data["result_analysis"] = analysis.to_dict()

                result.status = PhaseStatus.COMPLETED

        except Exception as e:
            result.status = PhaseStatus.FAILED
            result.errors.append(str(e))
            solution.add_error(str(e))
            solution.update_status(SolutionStatus.FAILED)

        result.completed_at = datetime.now()
        return result
