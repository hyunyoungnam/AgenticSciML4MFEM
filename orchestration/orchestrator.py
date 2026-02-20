"""
Main workflow orchestrator for AgenticSciML.

Coordinates all phases and manages the evolutionary search process.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base import AgentContext
from agents.llm.provider import LLMProvider, create_provider
from agents.roles.evaluator import EvaluatorAgent
from agents.roles.proposer import ProposerAgent
from agents.roles.critic import CriticAgent
from agents.roles.engineer import EngineerAgent
from agents.roles.debugger import DebuggerAgent
from agents.roles.result_analyst import ResultAnalystAgent
from agents.roles.claude_code_engineer import ClaudeCodeEngineer, ClaudeCodeEngineerConfig
from agents.roles.claude_code_debugger import ClaudeCodeDebugger, ClaudeCodeDebuggerConfig
from orchestration.phases import (
    Phase1AnalysisController,
    Phase2KnowledgeController,
    Phase3DebateController,
    Phase4ExecutionController,
    PhaseResult,
    PhaseStatus,
)
from evolution.solution import Solution, SolutionStatus, SolutionGenome
from evolution.tree import SolutionTree
from evolution.selection import ParentSelector, EnsembleSelector
from evaluation.pipeline import EvaluationPipeline
from evaluation.preflight import PreflightChecker
from evaluation.metrics import MetricsCalculator
from knowledge.failure_memory import FailureMemory


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestrator."""
    # Evolution settings
    max_generations: int = 20
    population_size: int = 10
    num_parents: int = 3

    # Debate settings
    num_debate_rounds: int = 4
    consensus_threshold: float = 0.7

    # Execution settings
    max_attempts: int = 3
    run_solver: bool = False
    solver_timeout: int = 3600

    # LLM settings
    openai_model: str = "gpt-4-turbo"
    anthropic_model: str = "claude-3-opus-20240229"

    # Claude Code settings
    use_claude_code: bool = True  # Use Claude Code for Engineer and Debugger
    claude_code_model: str = "sonnet"  # sonnet, opus, or haiku
    claude_code_max_turns: int = 25
    claude_code_timeout: int = 300

    # Paths
    output_dir: str = "outputs"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_generations": self.max_generations,
            "population_size": self.population_size,
            "num_parents": self.num_parents,
            "num_debate_rounds": self.num_debate_rounds,
            "consensus_threshold": self.consensus_threshold,
            "max_attempts": self.max_attempts,
            "run_solver": self.run_solver,
            "solver_timeout": self.solver_timeout,
            "openai_model": self.openai_model,
            "anthropic_model": self.anthropic_model,
            "use_claude_code": self.use_claude_code,
            "claude_code_model": self.claude_code_model,
            "claude_code_max_turns": self.claude_code_max_turns,
            "claude_code_timeout": self.claude_code_timeout,
            "output_dir": self.output_dir,
        }


@dataclass
class GenerationResult:
    """Result of a generation."""
    generation: int
    solutions: List[Solution]
    best_solution_id: Optional[str] = None
    best_score: float = 0.0
    num_converged: int = 0
    num_failed: int = 0
    phase_results: Dict[str, List[PhaseResult]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "solution_ids": [s.id for s in self.solutions],
            "best_solution_id": self.best_solution_id,
            "best_score": self.best_score,
            "num_converged": self.num_converged,
            "num_failed": self.num_failed,
        }


class AgenticOrchestrator:
    """
    Main orchestrator for the AgenticSciML framework.

    Coordinates:
    - Agent initialization and LLM provider setup
    - Four-phase workflow execution
    - Evolutionary tree management
    - Generation progression
    """

    def __init__(
        self,
        config: Optional[OrchestrationConfig] = None,
        openai_provider: Optional[LLMProvider] = None,
        anthropic_provider: Optional[LLMProvider] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Orchestration configuration
            openai_provider: OpenAI LLM provider
            anthropic_provider: Anthropic LLM provider
        """
        self.config = config or OrchestrationConfig()

        # Initialize LLM providers
        self.openai_provider = openai_provider or create_provider("openai")
        self.anthropic_provider = anthropic_provider or create_provider("anthropic")

        # Initialize agents
        self._init_agents()

        # Initialize evolution components
        self.solution_tree = SolutionTree(max_generations=self.config.max_generations)
        self.parent_selector = EnsembleSelector()

        # Initialize failure memory (runtime-specific learning)
        # Note: Static FEA knowledge base removed - LLMs have this built-in
        self.failure_memory = FailureMemory()

        # Initialize phase controllers
        self._init_phases()

        # Session state
        self.context: Optional[AgentContext] = None
        self.base_inp_path: Optional[str] = None
        self.morphing_config_path: Optional[str] = None
        self.current_generation: int = 0

    def _init_agents(self) -> None:
        """Initialize all agents with their LLM providers."""
        # Agents using Anthropic (Claude)
        self.evaluator = EvaluatorAgent(model=self.config.anthropic_model)
        self.evaluator.set_llm_provider(self.anthropic_provider)

        self.critic = CriticAgent(model=self.config.anthropic_model)
        self.critic.set_llm_provider(self.anthropic_provider)

        # Agents using OpenAI (GPT-4)
        self.proposer = ProposerAgent(model=self.config.openai_model)
        self.proposer.set_llm_provider(self.openai_provider)

        self.result_analyst = ResultAnalystAgent(model=self.config.openai_model)
        self.result_analyst.set_llm_provider(self.openai_provider)

        # Engineer and Debugger: Use Claude Code or hand-coded agents
        if self.config.use_claude_code:
            # Use Claude Code-powered agents
            engineer_config = ClaudeCodeEngineerConfig(
                model=self.config.claude_code_model,
                max_turns=self.config.claude_code_max_turns,
                timeout=self.config.claude_code_timeout,
                working_dir=str(Path.cwd()),
            )
            self.engineer = ClaudeCodeEngineer(config=engineer_config)

            debugger_config = ClaudeCodeDebuggerConfig(
                model=self.config.claude_code_model,
                max_turns=self.config.claude_code_max_turns,
                timeout=self.config.claude_code_timeout,
                working_dir=str(Path.cwd()),
            )
            self.debugger = ClaudeCodeDebugger(config=debugger_config)
        else:
            # Use traditional hand-coded agents
            self.debugger = DebuggerAgent(model=self.config.anthropic_model)
            self.debugger.set_llm_provider(self.anthropic_provider)

            self.engineer = EngineerAgent(model=self.config.openai_model)
            self.engineer.set_llm_provider(self.openai_provider)

    def _init_phases(self) -> None:
        """Initialize phase controllers."""
        self.phase1 = Phase1AnalysisController(self.evaluator)

        self.phase2 = Phase2KnowledgeController(
            failure_memory=self.failure_memory,
        )

        self.phase3 = Phase3DebateController(
            proposer=self.proposer,
            critic=self.critic,
            num_rounds=self.config.num_debate_rounds,
        )

        self.phase4 = Phase4ExecutionController(
            engineer=self.engineer,
            debugger=self.debugger,
            result_analyst=self.result_analyst,
            evaluation_pipeline=EvaluationPipeline(
                preflight_checker=PreflightChecker(),
                metrics_calculator=MetricsCalculator(),
                run_solver=self.config.run_solver,
                solver_timeout=self.config.solver_timeout,
            ),
            max_attempts=self.config.max_attempts,
        )

    async def initialize(
        self,
        base_inp_path: str,
        morphing_config_path: Optional[str] = None,
    ) -> PhaseResult:
        """
        Initialize the orchestrator with a base model.

        Runs Phase 1 (Analysis & Evaluation Contract).

        Args:
            base_inp_path: Path to the base .inp file
            morphing_config_path: Optional path to morphing config

        Returns:
            PhaseResult from Phase 1
        """
        self.base_inp_path = base_inp_path
        self.morphing_config_path = morphing_config_path

        # Create context
        self.context = AgentContext(
            base_inp_path=base_inp_path,
            config=self.config.to_dict(),
        )

        # Extract model info
        try:
            from manager import AbaqusManager
            manager = AbaqusManager(base_inp_path)
            self.context.model_info = self.evaluator.create_model_info(manager)
        except Exception as e:
            self.context.model_info = {"error": str(e)}

        # Run Phase 1
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        phase1_result = await self.phase1.execute(
            context=self.context,
            output_dir=str(output_dir),
        )

        return phase1_result

    async def run_generation(self, generation: int) -> GenerationResult:
        """
        Run a single generation of evolution.

        Args:
            generation: Generation number

        Returns:
            GenerationResult
        """
        if self.context is None:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        self.current_generation = generation
        gen_result = GenerationResult(generation=generation, solutions=[])
        gen_result.phase_results = {
            "phase2": [],
            "phase3": [],
            "phase4": [],
        }

        # Get parents for this generation
        if generation == 0:
            parents = []  # No parents for generation 0
        else:
            eligible = self.solution_tree.get_eligible_parents(
                generation=generation - 1,
                min_score=0.3,
            )
            selection = self.parent_selector.select(
                candidates=eligible,
                n=self.config.num_parents,
            )
            parents = selection.selected

        # Generate solutions for this generation
        num_solutions = self.config.population_size
        solutions_to_process = []

        for i in range(num_solutions):
            # Create solution
            if generation == 0 or not parents:
                # Initial generation
                solution = Solution(
                    generation=generation,
                    genome=SolutionGenome(),
                )
            else:
                # Select parent
                parent = parents[i % len(parents)]
                solution = self.solution_tree.create_child(
                    parent_id=parent.id,
                    genome=SolutionGenome(),  # Will be filled by debate
                )

            self.solution_tree.add_solution(solution)
            solutions_to_process.append(solution)

        # Process each solution through phases
        for solution in solutions_to_process:
            # Phase 2: Knowledge Funnel
            phase2_result = await self.phase2.execute(
                context=self.context,
                query=f"morphing delta_R generation {generation}",
                problem_type="2D_hole",
            )
            gen_result.phase_results["phase2"].append(phase2_result)

            # Phase 3: Debate
            parent = self.solution_tree.get_parent(solution.id)
            parent_mutations = parent.genome.get_mutation_summary() if parent else "Base model"

            phase3_result = await self.phase3.execute(
                context=self.context,
                solution=solution,
                parent_mutations=parent_mutations,
            )
            gen_result.phase_results["phase3"].append(phase3_result)

            # Skip Phase 4 if rejected in debate
            if solution.status == SolutionStatus.REJECTED:
                gen_result.num_failed += 1
                gen_result.solutions.append(solution)
                continue

            # Phase 4: Execution
            phase4_result = await self.phase4.execute(
                context=self.context,
                solution=solution,
                base_inp_path=self.base_inp_path,
                output_dir=str(Path(self.config.output_dir) / f"gen_{generation}"),
                config_path=self.morphing_config_path,
                run_solver=self.config.run_solver,
            )
            gen_result.phase_results["phase4"].append(phase4_result)

            # Update failure memory if failed
            if solution.status == SolutionStatus.FAILED:
                self.failure_memory.add_failure(
                    solution_id=solution.id,
                    delta_R=solution.genome.delta_R,
                    error="; ".join(solution.error_messages),
                    generation=generation,
                )
                gen_result.num_failed += 1
            else:
                gen_result.num_converged += 1

            gen_result.solutions.append(solution)

        # Calculate generation statistics
        scores = [
            s.metrics.compute_overall_score()
            for s in gen_result.solutions
            if s.is_successful()
        ]

        if scores:
            gen_result.best_score = max(scores)
            best_solution = max(
                [s for s in gen_result.solutions if s.is_successful()],
                key=lambda s: s.metrics.compute_overall_score(),
            )
            gen_result.best_solution_id = best_solution.id

        return gen_result

    async def run(
        self,
        base_inp_path: str,
        morphing_config_path: Optional[str] = None,
        num_generations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the full evolutionary process.

        Args:
            base_inp_path: Path to base .inp file
            morphing_config_path: Path to morphing config
            num_generations: Number of generations (default: config value)

        Returns:
            Summary of the run
        """
        num_generations = num_generations or self.config.max_generations

        # Initialize
        init_result = await self.initialize(base_inp_path, morphing_config_path)

        if init_result.status == PhaseStatus.FAILED:
            return {
                "status": "failed",
                "error": "Initialization failed",
                "phase1_result": init_result.to_dict(),
            }

        # Run generations
        generation_results = []

        for gen in range(num_generations):
            gen_result = await self.run_generation(gen)
            generation_results.append(gen_result.to_dict())

            # Check termination conditions
            if gen_result.num_converged == 0 and gen > 2:
                # No progress for too long
                break

        # Get best overall solution
        best_solutions = self.solution_tree.get_best_solutions(n=5)

        # Save tree
        tree_path = Path(self.config.output_dir) / "solution_tree.json"
        self.solution_tree.save_to_json(str(tree_path))

        # Save failure memory
        failure_path = Path(self.config.output_dir) / "failure_memory.json"
        self.failure_memory.save_to_json(str(failure_path))

        return {
            "status": "completed",
            "total_generations": len(generation_results),
            "total_solutions": len(self.solution_tree),
            "best_solutions": [s.to_dict() for s in best_solutions],
            "generation_results": generation_results,
            "tree_path": str(tree_path),
            "failure_memory_path": str(failure_path),
        }

    def get_solution(self, solution_id: str) -> Optional[Solution]:
        """Get a solution by ID."""
        return self.solution_tree.get_solution(solution_id)

    def get_best_solutions(self, n: int = 5) -> List[Solution]:
        """Get the top N solutions."""
        return self.solution_tree.get_best_solutions(n)

    def save_state(self, path: str) -> None:
        """Save orchestrator state to file."""
        state = {
            "config": self.config.to_dict(),
            "current_generation": self.current_generation,
            "base_inp_path": self.base_inp_path,
            "morphing_config_path": self.morphing_config_path,
            "context": self.context.to_dict() if self.context else None,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # Also save tree
        tree_path = Path(path).parent / "solution_tree.json"
        self.solution_tree.save_to_json(str(tree_path))
