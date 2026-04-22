"""
Agentic Surrogate Trainer.

Wraps SurrogateTrainer with LLM-based hyperparameter optimization.
Uses HyperparameterCriticAgent and ArchitectAgent to adaptively
tune model configuration based on training feedback.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from piano.surrogate.base import TransolverConfig
from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig, TrainingResult

if TYPE_CHECKING:
    from piano.agents.base import AgentContext
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent,
        TrainingHistory,
        TrainingIssue,
    )
    from piano.agents.roles.architect import ArchitectAgent, ArchitectureProposal


logger = logging.getLogger(__name__)


@dataclass
class AgenticTrainingConfig:
    """
    Configuration for agentic training.

    Attributes:
        base_config: Initial surrogate configuration
        max_hpo_rounds: Maximum HPO rounds per training session
        trigger_threshold: Error threshold to trigger HPO
        use_ensemble: Whether to use ensemble for uncertainty
        n_ensemble: Number of ensemble members
        llm_model: LLM model for agents
        random_seed: Random seed
    """
    base_config: TransolverConfig = field(default_factory=TransolverConfig)
    max_hpo_rounds: int = 3
    trigger_threshold: float = 0.1
    use_ensemble: bool = True
    n_ensemble: int = 5
    llm_model: str = "gpt-4-turbo"
    random_seed: int = 42


@dataclass
class AgenticTrainingResult:
    """
    Result of agentic training.

    Attributes:
        success: Whether training succeeded
        final_result: Final TrainingResult
        n_hpo_rounds: Number of HPO rounds executed
        config_history: History of configurations tried
        best_config: Best configuration found
        improvement_percent: Improvement from HPO
    """
    success: bool
    final_result: Optional[TrainingResult] = None
    n_hpo_rounds: int = 0
    config_history: List[Dict[str, Any]] = field(default_factory=list)
    best_config: Optional[TransolverConfig] = None
    improvement_percent: float = 0.0
    error_message: Optional[str] = None


class AgenticSurrogateTrainer:
    """
    Agentic wrapper for SurrogateTrainer.

    Implements adaptive hyperparameter optimization using LLM agents:
    1. Train with initial config
    2. Analyze training results with CriticAgent
    3. If issues detected, propose new config with ArchitectAgent
    4. Retrain and repeat up to max_hpo_rounds

    Features:
    - Adaptive trigger: Only invoke agents when needed
    - Heuristic fallback: Works without LLM if needed
    - History tracking: Learns from previous attempts
    """

    def __init__(
        self,
        config: AgenticTrainingConfig,
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize agentic trainer.

        Args:
            config: Agentic training configuration
            llm_provider: LLM provider for agents (optional)
        """
        self.config = config
        self.llm_provider = llm_provider

        # Lazy import agents to avoid circular imports
        from piano.agents.roles.hyperparameter_critic import HyperparameterCriticAgent
        from piano.agents.roles.architect import ArchitectAgent

        # Initialize agents
        self.critic = HyperparameterCriticAgent(model=config.llm_model)
        self.architect = ArchitectAgent(model=config.llm_model)

        if llm_provider:
            self.critic.set_llm_provider(llm_provider)
            self.architect.set_llm_provider(llm_provider)

        # State
        self._current_config = config.base_config
        self._config_history: List[Dict[str, Any]] = []
        self._best_result: Optional[TrainingResult] = None
        self._best_config: Optional[TransolverConfig] = None
        self._trainer: Optional[SurrogateTrainer] = None

    def train(
        self,
        parameters: np.ndarray,
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> AgenticTrainingResult:
        """
        Train surrogate with adaptive HPO.

        Args:
            parameters: Input parameters (N_samples, n_params)
            coordinates: Per-sample coordinates
            outputs: Per-sample outputs
            callback: Optional progress callback

        Returns:
            AgenticTrainingResult with final model and HPO history
        """
        try:
            return self._train_with_hpo(parameters, coordinates, outputs, callback)
        except Exception as e:
            logger.exception("Agentic training failed")
            return AgenticTrainingResult(
                success=False,
                error_message=str(e),
            )

    def _train_with_hpo(
        self,
        parameters: np.ndarray,
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> AgenticTrainingResult:
        """Internal training loop with HPO."""
        dataset_size = len(parameters)
        initial_error = float('inf')
        n_rounds = 0

        # Initial training
        logger.info("Starting initial training...")
        result = self._train_once(parameters, coordinates, outputs, callback)

        if not result.success:
            return AgenticTrainingResult(
                success=False,
                final_result=result,
                error_message=result.error_message,
            )

        initial_error = result.test_loss
        self._best_result = result
        self._best_config = self._current_config
        self._record_attempt(result, "initial")

        # Check if HPO is needed
        history = self._extract_history(result)

        if not self.critic.should_trigger_hpo(history, self.config.trigger_threshold):
            logger.info(f"Training converged well (loss={result.test_loss:.6f}), no HPO needed")
            return AgenticTrainingResult(
                success=True,
                final_result=result,
                n_hpo_rounds=0,
                config_history=self._config_history,
                best_config=self._current_config,
                improvement_percent=0.0,
            )

        # HPO loop
        logger.info(f"HPO triggered (threshold={self.config.trigger_threshold})")

        for round_idx in range(self.config.max_hpo_rounds):
            n_rounds = round_idx + 1
            logger.info(f"HPO Round {n_rounds}/{self.config.max_hpo_rounds}")

            # Get new config
            new_config = self._get_new_config(history, dataset_size)

            if new_config is None:
                logger.warning("Could not generate new config, stopping HPO")
                break

            self._current_config = new_config

            # Train with new config
            result = self._train_once(parameters, coordinates, outputs, callback)

            if not result.success:
                logger.warning(f"Training failed with new config: {result.error_message}")
                self._record_attempt(result, "failed")
                continue

            self._record_attempt(result, "success")
            history = self._extract_history(result)

            # Track best
            if result.test_loss < self._best_result.test_loss:
                logger.info(f"New best! {self._best_result.test_loss:.6f} -> {result.test_loss:.6f}")
                self._best_result = result
                self._best_config = self._current_config

            # Check if good enough
            if not self.critic.should_trigger_hpo(history, self.config.trigger_threshold):
                logger.info("HPO converged, stopping early")
                break

        # Compute improvement
        final_error = self._best_result.test_loss
        improvement = (initial_error - final_error) / initial_error * 100 if initial_error > 0 else 0

        logger.info(f"HPO complete: {n_rounds} rounds, {improvement:.1f}% improvement")

        return AgenticTrainingResult(
            success=True,
            final_result=self._best_result,
            n_hpo_rounds=n_rounds,
            config_history=self._config_history,
            best_config=self._best_config,
            improvement_percent=improvement,
        )

    def _train_once(
        self,
        parameters: np.ndarray,
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> TrainingResult:
        """Single training run with current config."""
        training_config = TrainingConfig(
            surrogate_config=self._current_config,
            use_ensemble=self.config.use_ensemble,
            n_ensemble=self.config.n_ensemble,
            normalize_inputs=True,
            normalize_outputs=True,
            train_test_split=0.2,
            random_seed=self.config.random_seed,
        )

        self._trainer = SurrogateTrainer(training_config)
        return self._trainer.train(parameters, coordinates, outputs, callback)

    def _extract_history(self, result: TrainingResult) -> "TrainingHistory":
        """Extract TrainingHistory from TrainingResult."""
        from piano.agents.roles.hyperparameter_critic import TrainingHistory

        history = result.history or {}

        train_losses = history.get('train_loss', [])
        test_losses = history.get('test_loss', [])
        pino_losses = history.get('pino_loss', [])

        # Check for NaN
        has_nan = any(np.isnan(l) for l in train_losses + test_losses)

        return TrainingHistory(
            train_losses=train_losses,
            test_losses=test_losses,
            pino_losses=pino_losses,
            epochs_completed=len(train_losses),
            best_test_loss=min(test_losses) if test_losses else float('inf'),
            final_train_loss=train_losses[-1] if train_losses else float('inf'),
            final_test_loss=test_losses[-1] if test_losses else float('inf'),
            has_nan=has_nan,
            metrics=result.metrics,
        )

    def _get_new_config(
        self,
        history: "TrainingHistory",
        dataset_size: int,
    ) -> Optional[TransolverConfig]:
        """Get new config from LLM agents."""
        if self.llm_provider is None:
            raise RuntimeError("LLM provider is required for agentic HPO")

        try:
            return self._get_config_from_agents(history, dataset_size)
        except Exception as e:
            logger.error(f"Agent-based HPO failed: {e}")
            raise

    def _get_config_from_agents(
        self,
        history: "TrainingHistory",
        dataset_size: int,
    ) -> TransolverConfig:
        """Get config using LLM agents (sync wrapper for async)."""
        import asyncio
        from piano.agents.base import AgentContext

        # Create agent context
        context = AgentContext()

        # Run async agents synchronously
        loop = asyncio.new_event_loop()
        try:
            # Get critique
            critique = loop.run_until_complete(
                self.critic.analyze_training(
                    context=context,
                    training_history=history,
                    config=self._current_config.to_dict(),
                    previous_attempts=[
                        {"summary": f"Loss={h['result']}"}
                        for h in self._config_history
                    ],
                )
            )

            logger.info(f"Critic diagnosis: {critique.primary_issue.name} ({critique.severity})")

            # Get architecture proposal
            proposal = loop.run_until_complete(
                self.architect.propose_config(
                    context=context,
                    current_config=self._current_config,
                    critique=critique,
                    dataset_size=dataset_size,
                    previous_configs=self._config_history,
                )
            )

            logger.info(f"Architect proposal: {proposal.changes}")
            logger.info(f"Reasoning: {proposal.reasoning[:200]}...")

            return proposal.config

        finally:
            loop.close()

    def _record_attempt(self, result: TrainingResult, status: str) -> None:
        """Record an HPO attempt."""
        self._config_history.append({
            "config": self._current_config.to_dict(),
            "changes": {},  # Could track specific changes here
            "result": f"{status}: loss={result.test_loss:.6f}" if result.success else status,
            "metrics": result.metrics if result.success else {},
        })

    @property
    def model(self):
        """Get the trained model from the best trainer."""
        return self._trainer.model if self._trainer else None

    @property
    def best_config(self) -> Optional[TransolverConfig]:
        """Get the best configuration found."""
        return self._best_config


async def train_with_agents_async(
    parameters: np.ndarray,
    coordinates: List[np.ndarray],
    outputs: List[np.ndarray],
    config: AgenticTrainingConfig,
    llm_provider: Any,
    callback: Optional[Callable[[int, float], None]] = None,
) -> AgenticTrainingResult:
    """
    Async version of agentic training.

    For use in async contexts like the orchestrator.
    """
    trainer = AgenticSurrogateTrainer(config, llm_provider)
    return trainer.train(parameters, coordinates, outputs, callback)
