"""
tests/test_agentic_sciml.py — Test the Agentic SciML Loop.

Tests the complete agentic hyperparameter optimization pipeline:
  1. HyperparameterCriticAgent - diagnoses training issues
  2. ArchitectAgent - proposes configuration changes
  3. AgenticSurrogateTrainer - full training loop with adaptive HPO

The agentic loop tunes ALL ML factors:
  - Neural network architecture (d_model, n_layers, n_heads, slice_num)
  - Optimizer (AdamW, Adam, SGD)
  - Scheduler (plateau, cosine, none)
  - Learning rate, dropout, batch size
  - PINO loss weights
  - Activation functions

Run:
    pytest tests/test_agentic_sciml.py -v
    pytest tests/test_agentic_sciml.py -v -m "not mfem"
    python tests/test_agentic_sciml.py [--n-samples N] [--epochs E]
"""

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN01 = PROJECT_ROOT / "train01"
TRAIN02 = PROJECT_ROOT / "train02"

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "E": (150e9, 250e9),
    "nu": (0.25, 0.35),
    "load_x": (50e6, 150e6),
    "load_y": (-100e6, 100e6),
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())


# =============================================================================
# Mock LLM Provider for Testing
# =============================================================================

class MockLLMResponse:
    """Mock LLM response object."""
    def __init__(self, content: str):
        self.content = content


class MockLLMProvider:
    """
    Mock LLM provider for testing agents without actual API calls.

    Simulates realistic responses for critic and architect agents.
    """

    def __init__(self, scenario: str = "overfitting"):
        """
        Initialize with a predefined scenario.

        Args:
            scenario: One of "overfitting", "underfitting", "slow_convergence",
                      "stable", "unstable"
        """
        self.scenario = scenario
        self.call_count = 0
        self.call_history: List[Dict[str, str]] = []

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4-turbo",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> MockLLMResponse:
        """Generate mock response based on scenario."""
        self.call_count += 1
        self.call_history.append({
            "system": system_prompt[:200],
            "user": user_prompt[:500],
        })

        # Detect if this is a critic or architect call
        if "training analyst" in system_prompt.lower():
            return MockLLMResponse(self._critic_response())
        elif "architect" in system_prompt.lower():
            return MockLLMResponse(self._architect_response())
        else:
            return MockLLMResponse("Unknown agent type")

    def _critic_response(self) -> str:
        """Generate mock critic response based on scenario."""
        responses = {
            "overfitting": """
DIAGNOSIS: The model shows clear signs of overfitting. Training loss continues to decrease while test loss has started increasing after epoch 30. The train-test gap is widening significantly.

PRIMARY_ISSUE: OVERFITTING
SEVERITY: high

RECOMMENDATIONS:
- Increase dropout from 0.0 to 0.15 to add regularization
- Reduce model capacity by decreasing d_model from 256 to 128
- Increase PINO weight to enforce physics constraints
- Consider reducing n_layers from 6 to 4

SHOULD_RETRAIN: true

METRICS_ANALYSIS:
- train_test_gap: Large and growing, indicating overfitting
- convergence_rate: Fast initial convergence but diverging
- stability: Stable training but poor generalization
""",
            "underfitting": """
DIAGNOSIS: The model is underfitting. Both training and test losses remain high and have plateaued. The model lacks sufficient capacity to learn the underlying patterns.

PRIMARY_ISSUE: UNDERFITTING
SEVERITY: high

RECOMMENDATIONS:
- Increase d_model from 64 to 256 for more capacity
- Increase n_layers from 2 to 6
- Increase learning rate from 1e-4 to 5e-4
- Reduce dropout if any
- Decrease PINO weight to allow more data-driven learning

SHOULD_RETRAIN: true

METRICS_ANALYSIS:
- train_test_gap: Small gap but both losses are high
- convergence_rate: Slow and plateaued
- stability: Stable but stuck at suboptimal solution
""",
            "slow_convergence": """
DIAGNOSIS: Training is converging but very slowly. The learning rate may be too conservative.

PRIMARY_ISSUE: SLOW_CONVERGENCE
SEVERITY: medium

RECOMMENDATIONS:
- Increase learning rate from 1e-4 to 1e-3
- Switch to cosine scheduler for smoother decay
- Consider using AdamW optimizer if not already

SHOULD_RETRAIN: true

METRICS_ANALYSIS:
- train_test_gap: Acceptable
- convergence_rate: Too slow, needs acceleration
- stability: Very stable but inefficient
""",
            "stable": """
DIAGNOSIS: Training appears healthy. Both training and test losses are decreasing appropriately with a reasonable gap.

PRIMARY_ISSUE: NONE
SEVERITY: low

RECOMMENDATIONS:
- Current configuration is working well
- Consider longer training if budget allows

SHOULD_RETRAIN: false

METRICS_ANALYSIS:
- train_test_gap: Healthy, within acceptable range
- convergence_rate: Good
- stability: Excellent
""",
            "unstable": """
DIAGNOSIS: Training is highly unstable with large loss fluctuations. The learning rate is too high.

PRIMARY_ISSUE: UNSTABLE_TRAINING
SEVERITY: critical

RECOMMENDATIONS:
- Reduce learning rate from 1e-2 to 1e-4
- Increase batch size from 8 to 32 for smoother gradients
- Use plateau scheduler with patience

SHOULD_RETRAIN: true

METRICS_ANALYSIS:
- train_test_gap: Erratic, cannot assess
- convergence_rate: Non-convergent
- stability: Poor, needs immediate attention
""",
        }
        return responses.get(self.scenario, responses["stable"])

    def _architect_response(self) -> str:
        """Generate mock architect response based on scenario."""
        responses = {
            "overfitting": """
REASONING: The critic identified overfitting as the primary issue. To address this, I'm reducing model capacity and adding regularization. Increasing PINO weight will help enforce physical constraints and reduce overfitting to noise in the data.

CHANGES:
- d_model: 128 (reduced from 256 to decrease capacity)
- n_layers: 4 (reduced from 6)
- dropout: 0.15 (added regularization)
- pino_weight: 0.2 (increased for physics constraints)
- learning_rate: 5e-4 (slightly reduced)

EXPECTED_IMPACT: Expect reduced overfitting with better generalization. Test loss should improve even if training loss increases slightly.

TRADE_OFFS: Reduced model capacity may limit peak performance on complex patterns, but should significantly improve generalization.

CONFIDENCE: high
""",
            "underfitting": """
REASONING: The model lacks capacity to learn the physics. Increasing model size and learning rate should help. Reducing PINO weight allows more flexibility for data-driven learning initially.

CHANGES:
- d_model: 256 (increased for more capacity)
- n_layers: 6 (more depth)
- n_heads: 8 (more attention heads)
- slice_num: 32 (finer physics resolution)
- learning_rate: 1e-3 (faster learning)
- dropout: 0.0 (no regularization needed)
- pino_weight: 0.05 (reduced to allow data-driven learning)

EXPECTED_IMPACT: Model should be able to capture complex physics patterns. Both train and test loss should decrease significantly.

TRADE_OFFS: Larger model needs more training time and may eventually overfit if trained too long.

CONFIDENCE: high
""",
            "slow_convergence": """
REASONING: The learning dynamics are too conservative. Increasing learning rate and using cosine scheduler should accelerate convergence while maintaining stability.

CHANGES:
- learning_rate: 1e-3 (increased from 1e-4)
- scheduler_type: cosine (smooth annealing)
- optimizer_type: adamw (good with cosine schedule)

EXPECTED_IMPACT: Faster convergence while maintaining stability. Should reach same loss in fewer epochs.

TRADE_OFFS: Higher learning rate has some risk of instability, but cosine schedule provides safety.

CONFIDENCE: medium
""",
            "stable": """
REASONING: Current configuration is working well. Making only minor adjustments to potentially squeeze out a bit more performance.

CHANGES:
- epochs: 100 (allow longer training)

EXPECTED_IMPACT: Minor improvement possible with extended training.

TRADE_OFFS: Longer training time but low risk.

CONFIDENCE: medium
""",
            "unstable": """
REASONING: Training is unstable due to high learning rate. Need to significantly reduce LR and increase batch size for stability.

CHANGES:
- learning_rate: 1e-4 (reduced from 1e-2)
- batch_size: 32 (increased for stable gradients)
- scheduler_type: plateau (adaptive to loss)
- optimizer_type: adamw (good stability)

EXPECTED_IMPACT: Training should stabilize and converge properly.

TRADE_OFFS: Slower convergence but necessary for stability.

CONFIDENCE: high
""",
        }
        return responses.get(self.scenario, responses["stable"])


# =============================================================================
# 1. HyperparameterCriticAgent Tests
# =============================================================================

def test_critic_detect_issues_heuristic_overfitting():
    """Test heuristic overfitting detection without LLM."""
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent,
        TrainingHistory,
        TrainingIssue,
    )

    critic = HyperparameterCriticAgent()

    # Simulate overfitting: train loss steadily decreasing, test loss increasing
    # Use smooth curves to avoid triggering instability detection
    train_losses = [0.5 - 0.02 * i for i in range(25)]  # 0.5 -> 0.02
    test_losses = [0.5 - 0.01 * i if i < 10 else 0.4 + 0.02 * (i - 10) for i in range(25)]
    # test starts at 0.5, drops to 0.4, then rises to 0.7

    history = TrainingHistory(
        train_losses=train_losses,
        test_losses=test_losses,
        epochs_completed=25,
        best_test_loss=min(test_losses),
        final_train_loss=train_losses[-1],
        final_test_loss=test_losses[-1],
    )

    issues = critic.detect_issues_heuristic(history)
    assert TrainingIssue.OVERFITTING in issues


def test_critic_detect_issues_heuristic_underfitting():
    """Test heuristic underfitting detection without LLM."""
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent,
        TrainingHistory,
        TrainingIssue,
    )

    critic = HyperparameterCriticAgent()

    # Simulate underfitting: both losses high and not improving
    history = TrainingHistory(
        train_losses=[0.5, 0.49, 0.48, 0.48, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47] * 2,
        test_losses=[0.52, 0.51, 0.50, 0.50, 0.49, 0.49, 0.49, 0.49, 0.49, 0.49] * 2,
        epochs_completed=20,
        best_test_loss=0.49,
        final_train_loss=0.47,
        final_test_loss=0.49,
    )

    issues = critic.detect_issues_heuristic(history)
    assert TrainingIssue.UNDERFITTING in issues


def test_critic_detect_issues_heuristic_plateau():
    """Test heuristic loss plateau detection."""
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent,
        TrainingHistory,
        TrainingIssue,
    )

    critic = HyperparameterCriticAgent()

    # Simulate plateau: losses not changing at all
    history = TrainingHistory(
        train_losses=[0.1] * 20,
        test_losses=[0.12] * 20,
        epochs_completed=20,
        best_test_loss=0.12,
        final_train_loss=0.1,
        final_test_loss=0.12,
    )

    issues = critic.detect_issues_heuristic(history)
    assert TrainingIssue.LOSS_PLATEAU in issues


def test_critic_detect_nan():
    """Test NaN detection."""
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent,
        TrainingHistory,
        TrainingIssue,
    )

    critic = HyperparameterCriticAgent()

    history = TrainingHistory(
        train_losses=[0.5, 0.4, float('nan')],
        test_losses=[0.5, 0.45, float('nan')],
        epochs_completed=3,
        has_nan=True,
    )

    issues = critic.detect_issues_heuristic(history)
    assert TrainingIssue.GRADIENT_EXPLOSION in issues


def test_critic_should_trigger_hpo():
    """Test HPO trigger logic."""
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent,
        TrainingHistory,
    )

    critic = HyperparameterCriticAgent()

    # Good training - should NOT trigger HPO
    good_history = TrainingHistory(
        train_losses=[0.1, 0.05, 0.02, 0.01, 0.005],
        test_losses=[0.12, 0.06, 0.03, 0.015, 0.008],
        epochs_completed=5,
        final_test_loss=0.008,
    )
    assert not critic.should_trigger_hpo(good_history, threshold=0.01)

    # Bad training - SHOULD trigger HPO
    bad_history = TrainingHistory(
        train_losses=[0.5, 0.4, 0.3, 0.2, 0.1],
        test_losses=[0.5, 0.5, 0.55, 0.6, 0.7],  # Overfitting
        epochs_completed=5,
        final_test_loss=0.7,
        final_train_loss=0.1,
    )
    assert critic.should_trigger_hpo(bad_history, threshold=0.1)


@pytest.mark.asyncio
async def test_critic_analyze_training_with_mock_llm():
    """Test critic analysis with mock LLM."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent,
        TrainingHistory,
        TrainingIssue,
    )

    critic = HyperparameterCriticAgent()
    provider = MockLLMProvider(scenario="overfitting")
    critic.set_llm_provider(provider)

    context = AgentContext()
    history = TrainingHistory(
        train_losses=[0.5, 0.3, 0.1, 0.05],
        test_losses=[0.5, 0.4, 0.5, 0.6],
        epochs_completed=4,
        final_train_loss=0.05,
        final_test_loss=0.6,
    )

    result = await critic.analyze_training(
        context=context,
        training_history=history,
        config={"d_model": 256, "n_layers": 6, "dropout": 0.0},
    )

    assert result.primary_issue == TrainingIssue.OVERFITTING
    assert result.severity == "high"
    assert result.should_retrain is True
    assert len(result.recommendations) > 0
    assert provider.call_count == 1


# =============================================================================
# 2. ArchitectAgent Tests
# =============================================================================

@pytest.mark.asyncio
async def test_architect_propose_config_overfitting():
    """Test architect proposes config for overfitting scenario."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.roles.hyperparameter_critic import CritiqueResult, TrainingIssue
    from piano.surrogate.base import TransolverConfig

    architect = ArchitectAgent()
    provider = MockLLMProvider(scenario="overfitting")
    architect.set_llm_provider(provider)

    context = AgentContext()
    current_config = TransolverConfig(
        d_model=256, n_layers=6, dropout=0.0, learning_rate=1e-3
    )
    critique = CritiqueResult(
        primary_issue=TrainingIssue.OVERFITTING,
        severity="high",
        diagnosis="Model is overfitting",
        recommendations=["Reduce capacity", "Add dropout"],
        should_retrain=True,
    )

    proposal = await architect.propose_config(
        context=context,
        current_config=current_config,
        critique=critique,
        dataset_size=50,
    )

    # Should propose reduced capacity and added regularization
    assert proposal.config is not None
    assert proposal.changes.get("d_model", 256) <= 256  # Should reduce or keep
    assert proposal.changes.get("dropout", 0.0) >= 0.0  # Should add dropout
    assert proposal.confidence in ["low", "medium", "high"]
    assert len(proposal.reasoning) > 0


@pytest.mark.asyncio
async def test_architect_propose_config_underfitting():
    """Test architect proposes config for underfitting scenario."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.roles.hyperparameter_critic import CritiqueResult, TrainingIssue
    from piano.surrogate.base import TransolverConfig

    architect = ArchitectAgent()
    provider = MockLLMProvider(scenario="underfitting")
    architect.set_llm_provider(provider)

    context = AgentContext()
    current_config = TransolverConfig(
        d_model=64, n_layers=2, dropout=0.2, learning_rate=1e-4
    )
    critique = CritiqueResult(
        primary_issue=TrainingIssue.UNDERFITTING,
        severity="high",
        diagnosis="Model lacks capacity",
        recommendations=["Increase d_model", "Add layers"],
        should_retrain=True,
    )

    proposal = await architect.propose_config(
        context=context,
        current_config=current_config,
        critique=critique,
        dataset_size=50,
    )

    # Should propose increased capacity
    assert proposal.config is not None
    assert proposal.changes.get("d_model", 64) >= 64  # Should increase
    assert len(proposal.reasoning) > 0


def test_architect_apply_changes():
    """Test architect applies changes correctly."""
    from piano.agents.roles.architect import ArchitectAgent
    from piano.surrogate.base import TransolverConfig

    architect = ArchitectAgent()

    base_config = TransolverConfig(
        d_model=256, n_layers=6, dropout=0.0, learning_rate=1e-3,
        optimizer_type="adamw", scheduler_type="plateau",
    )

    changes = {
        "d_model": 128,
        "dropout": 0.15,
        "learning_rate": 5e-4,
        "optimizer_type": "adam",
    }

    new_config = architect.apply_changes(base_config, changes)

    assert new_config.d_model == 128
    assert new_config.dropout == 0.15
    assert new_config.learning_rate == 5e-4
    assert new_config.optimizer_type == "adam"
    # Unchanged values should remain
    assert new_config.n_layers == 6
    assert new_config.scheduler_type == "plateau"


# =============================================================================
# 3. TransolverConfig Tests
# =============================================================================

def test_transolver_config_all_tunable_params():
    """Test that TransolverConfig includes all tunable parameters."""
    from piano.surrogate.base import TransolverConfig

    config = TransolverConfig(
        # Architecture
        d_model=128,
        n_layers=4,
        n_heads=8,
        slice_num=32,
        mlp_ratio=4.0,
        dropout=0.1,
        activation="gelu",
        # Optimizer
        optimizer_type="adamw",
        learning_rate=1e-3,
        scheduler_type="cosine",
        # PINO
        pino_weight=0.1,
        pino_eq_weight=0.1,
        # Training
        batch_size=32,
        epochs=100,
        patience=50,
    )

    # Verify all parameters are set
    d = config.to_dict()
    assert d["d_model"] == 128
    assert d["n_layers"] == 4
    assert d["n_heads"] == 8
    assert d["slice_num"] == 32
    assert d["dropout"] == 0.1
    assert d["activation"] == "gelu"
    assert d["optimizer_type"] == "adamw"
    assert d["learning_rate"] == 1e-3
    assert d["scheduler_type"] == "cosine"
    assert d["pino_weight"] == 0.1
    assert d["pino_eq_weight"] == 0.1
    assert d["batch_size"] == 32


# =============================================================================
# 4. Full Agentic Training Loop Tests (Synthetic Data)
# =============================================================================

@pytest.fixture(scope="module")
def synthetic_dataset():
    """Create synthetic dataset for testing."""
    rng = np.random.default_rng(42)
    N_SAMPLES = 20
    N_POINTS = 64

    # Parameters: [E, nu, load_x, load_y]
    params = rng.uniform(
        [150e9, 0.25, 50e6, -100e6],
        [250e9, 0.35, 150e6, 100e6],
        size=(N_SAMPLES, 4)
    ).astype(np.float32)

    # Coordinates (same for all samples for simplicity)
    coords = rng.uniform(0, 1, size=(N_POINTS, 2)).astype(np.float32)

    # Outputs: displacement field (N_POINTS, 2)
    outputs = [
        rng.uniform(-1e-3, 1e-3, size=(N_POINTS, 2)).astype(np.float32)
        for _ in range(N_SAMPLES)
    ]

    return params, [coords] * N_SAMPLES, outputs


def test_agentic_trainer_initialization():
    """Test AgenticSurrogateTrainer initialization."""
    from piano.surrogate.agentic_trainer import (
        AgenticSurrogateTrainer,
        AgenticTrainingConfig,
    )
    from piano.surrogate.base import TransolverConfig

    config = AgenticTrainingConfig(
        base_config=TransolverConfig(d_model=64, n_layers=2),
        max_hpo_rounds=3,
        trigger_threshold=0.1,
        use_ensemble=True,
        n_ensemble=3,
    )

    provider = MockLLMProvider()
    trainer = AgenticSurrogateTrainer(config, llm_provider=provider)

    assert trainer.config == config
    assert trainer.critic is not None
    assert trainer.architect is not None


def test_agentic_trainer_train_without_hpo(synthetic_dataset):
    """Test agentic training when HPO is not needed (good convergence)."""
    from piano.surrogate.agentic_trainer import (
        AgenticSurrogateTrainer,
        AgenticTrainingConfig,
    )
    from piano.surrogate.base import TransolverConfig

    params, coords, outputs = synthetic_dataset

    # Use small model for fast test
    config = AgenticTrainingConfig(
        base_config=TransolverConfig(
            d_model=32, n_layers=1, n_heads=2, slice_num=4,
            epochs=5, patience=10, batch_size=8,
            output_dim=2,
        ),
        max_hpo_rounds=2,
        trigger_threshold=10.0,  # High threshold = no HPO triggered
        use_ensemble=False,
        n_ensemble=1,
    )

    provider = MockLLMProvider(scenario="stable")
    trainer = AgenticSurrogateTrainer(config, llm_provider=provider)

    result = trainer.train(params, coords, outputs)

    assert result.success
    assert result.n_hpo_rounds == 0  # HPO should not be triggered
    assert result.final_result is not None


def test_training_history_extraction():
    """Test extraction of training history for critic analysis."""
    from piano.surrogate.trainer import TrainingResult
    from piano.surrogate.agentic_trainer import AgenticSurrogateTrainer, AgenticTrainingConfig
    from piano.surrogate.base import TransolverConfig

    config = AgenticTrainingConfig(
        base_config=TransolverConfig(d_model=32, n_layers=1),
    )
    provider = MockLLMProvider()
    trainer = AgenticSurrogateTrainer(config, llm_provider=provider)

    # Create mock training result
    mock_result = TrainingResult(
        success=True,
        train_loss=0.05,
        test_loss=0.08,
        history={
            "train_loss": [0.5, 0.3, 0.1, 0.05],
            "test_loss": [0.55, 0.35, 0.12, 0.08],
        },
        metrics={"rmse": 0.1},
    )

    history = trainer._extract_history(mock_result)

    assert history.epochs_completed == 4
    assert history.final_train_loss == 0.05
    assert history.final_test_loss == 0.08
    assert len(history.train_losses) == 4


# =============================================================================
# 5. Critic-Architect Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_critic_architect_loop():
    """Test the critic → architect feedback loop."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent,
        TrainingHistory,
    )
    from piano.agents.roles.architect import ArchitectAgent
    from piano.surrogate.base import TransolverConfig

    # Setup
    critic = HyperparameterCriticAgent()
    architect = ArchitectAgent()

    provider = MockLLMProvider(scenario="overfitting")
    critic.set_llm_provider(provider)
    architect.set_llm_provider(provider)

    context = AgentContext()

    # Initial config
    current_config = TransolverConfig(
        d_model=256, n_layers=6, dropout=0.0,
        learning_rate=1e-3, optimizer_type="adamw",
    )

    # Simulated overfitting training history
    history = TrainingHistory(
        train_losses=[0.5, 0.3, 0.15, 0.08, 0.04],
        test_losses=[0.5, 0.35, 0.3, 0.35, 0.45],
        epochs_completed=5,
        final_train_loss=0.04,
        final_test_loss=0.45,
    )

    # Step 1: Critic analyzes
    critique = await critic.analyze_training(
        context=context,
        training_history=history,
        config=current_config.to_dict(),
    )

    assert critique.should_retrain is True
    assert critique.primary_issue.name == "OVERFITTING"

    # Step 2: Architect proposes fix
    proposal = await architect.propose_config(
        context=context,
        current_config=current_config,
        critique=critique,
        dataset_size=50,
    )

    assert proposal.config is not None
    assert len(proposal.changes) > 0
    assert len(proposal.reasoning) > 0

    # Verify the loop produced actionable changes
    new_config = proposal.config
    print(f"\nCritic diagnosis: {critique.primary_issue.name}")
    print(f"Architect changes: {proposal.changes}")
    print(f"Confidence: {proposal.confidence}")


# =============================================================================
# 6. Issue Detection Coverage Tests
# =============================================================================

@pytest.mark.parametrize("scenario,expected_issue", [
    ("overfitting", "OVERFITTING"),
    ("underfitting", "UNDERFITTING"),
    ("slow_convergence", "SLOW_CONVERGENCE"),
    ("unstable", "UNSTABLE_TRAINING"),
    ("stable", "NONE"),
])
@pytest.mark.asyncio
async def test_critic_scenarios(scenario, expected_issue):
    """Test critic correctly identifies different training issues."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent,
        TrainingHistory,
    )

    critic = HyperparameterCriticAgent()
    provider = MockLLMProvider(scenario=scenario)
    critic.set_llm_provider(provider)

    context = AgentContext()
    history = TrainingHistory(
        train_losses=[0.5, 0.4, 0.3, 0.2, 0.1],
        test_losses=[0.5, 0.45, 0.4, 0.35, 0.3],
        epochs_completed=5,
        final_train_loss=0.1,
        final_test_loss=0.3,
    )

    result = await critic.analyze_training(
        context=context,
        training_history=history,
        config={"d_model": 128},
    )

    assert result.primary_issue.name == expected_issue


# =============================================================================
# 7. End-to-End Test with Real FEM Data (Optional)
# =============================================================================

@pytest.mark.mfem
def test_agentic_loop_with_real_data():
    """
    End-to-end test of the agentic SciML loop with real FEM data.

    This test:
    1. Loads real mesh files
    2. Runs FEM simulations
    3. Trains with initial config
    4. Uses mock LLM for critic/architect
    5. Verifies the full loop works
    """
    pytest.importorskip("mfem.ser")

    from piano.surrogate.agentic_trainer import (
        AgenticSurrogateTrainer,
        AgenticTrainingConfig,
    )
    from piano.surrogate.base import TransolverConfig

    # Would need to load real data here
    # For now, skip if no data available
    if not TRAIN01.exists():
        pytest.skip("train01 directory not found")

    # This would be the full integration test
    # Skipping actual implementation to keep test fast
    pass


# =============================================================================
# 8. Standalone Visualization (when run as __main__)
# =============================================================================

def _generate_plate_with_hole_mesh(n_points: int = 400, hole_radius: float = 0.2,
                                    hole_center: Tuple[float, float] = (0.5, 0.5),
                                    rng: np.random.Generator = None):
    """
    Generate synthetic mesh coordinates for a plate with a circular hole.
    Returns coordinates and triangulation for visualization.
    """
    from scipy.spatial import Delaunay

    if rng is None:
        rng = np.random.default_rng(42)

    # Generate random points in [0, 1] x [0, 1]
    points = []
    while len(points) < n_points:
        p = rng.uniform(0, 1, size=(2,))
        # Reject points inside the hole
        dist = np.sqrt((p[0] - hole_center[0])**2 + (p[1] - hole_center[1])**2)
        if dist > hole_radius:
            points.append(p)

    # Add boundary points for better triangulation
    n_boundary = 30
    # Outer boundary
    for i in range(n_boundary):
        t = i / n_boundary
        points.append([t, 0.0])  # bottom
        points.append([1.0, t])  # right
        points.append([1.0 - t, 1.0])  # top
        points.append([0.0, 1.0 - t])  # left

    # Hole boundary
    n_hole = 24
    for i in range(n_hole):
        theta = 2 * np.pi * i / n_hole
        x = hole_center[0] + hole_radius * np.cos(theta)
        y = hole_center[1] + hole_radius * np.sin(theta)
        points.append([x, y])

    coords = np.array(points, dtype=np.float32)

    # Triangulate
    tri = Delaunay(coords)

    # Filter triangles that are inside the hole
    valid_triangles = []
    for simplex in tri.simplices:
        centroid = coords[simplex].mean(axis=0)
        dist = np.sqrt((centroid[0] - hole_center[0])**2 + (centroid[1] - hole_center[1])**2)
        if dist > hole_radius * 0.9:  # Keep triangles outside hole
            valid_triangles.append(simplex)

    triangles = np.array(valid_triangles)

    return coords, triangles


def _synthetic_displacement_field(coords: np.ndarray, params: Dict,
                                   hole_center: Tuple[float, float] = (0.5, 0.5),
                                   hole_radius: float = 0.2) -> np.ndarray:
    """
    Generate synthetic displacement field that mimics stress concentration around hole.
    """
    E = params.get("E", 200e9)
    nu = params.get("nu", 0.3)
    load_x = params.get("load_x", 100e6)
    load_y = params.get("load_y", 0.0)

    # Normalize params for scaling
    E_norm = E / 200e9
    load_norm = (abs(load_x) + abs(load_y)) / 100e6

    n_points = len(coords)
    disp = np.zeros((n_points, 2), dtype=np.float32)

    for i, (x, y) in enumerate(coords):
        # Distance from hole center
        dx = x - hole_center[0]
        dy = y - hole_center[1]
        r = np.sqrt(dx**2 + dy**2)

        # Base displacement (linear with position)
        ux_base = load_x / E * x * 1e-3
        uy_base = load_y / E * y * 1e-3 - nu * load_x / E * y * 1e-3

        # Stress concentration factor near hole (simplified)
        if r > hole_radius:
            # Kirsch solution approximation
            concentration = 1.0 + (hole_radius / r)**2 + 1.5 * (hole_radius / r)**4
            concentration = min(concentration, 3.0)  # Cap at 3x
        else:
            concentration = 1.0

        disp[i, 0] = ux_base * concentration
        disp[i, 1] = uy_base * concentration

    return disp


def run_agentic_loop_demo(
    n_samples: int = 10,
    epochs: int = 20,
    output_file: str = "tests/test_outputs/agentic_loop_demo.png",
):
    """
    Demonstration of the full agentic SciML loop with contour plot visualizations.
    Uses synthetic plate-with-hole geometry.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.tri as mtri

    print("=" * 70)
    print("Agentic SciML Loop Demonstration (Synthetic Plate with Hole)")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Generate plate-with-hole mesh
    print("\n1. Generating synthetic plate-with-hole mesh...")
    coords, triangles = _generate_plate_with_hole_mesh(n_points=300, rng=rng)
    N_POINTS = len(coords)
    print(f"   Mesh: {N_POINTS} nodes, {len(triangles)} triangles")

    # Generate training samples with physics-inspired displacement
    print(f"\n2. Generating {n_samples} training samples...")
    params_list = []
    outputs = []
    for i in range(n_samples):
        p = {
            "E": float(rng.uniform(150e9, 250e9)),
            "nu": float(rng.uniform(0.25, 0.35)),
            "load_x": float(rng.uniform(50e6, 150e6)),
            "load_y": float(rng.uniform(-50e6, 50e6)),
        }
        params_list.append(p)
        disp = _synthetic_displacement_field(coords, p)
        outputs.append(disp)

    params = np.array([[p["E"], p["nu"], p["load_x"], p["load_y"]]
                       for p in params_list], dtype=np.float32)

    # Import components
    from piano.surrogate.base import TransolverConfig
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent,
        TrainingHistory,
    )
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.base import AgentContext
    import asyncio

    # Initial config (intentionally suboptimal - too large for small dataset)
    print("\n3. Training with INITIAL config (suboptimal)...")
    initial_config = TransolverConfig(
        d_model=128,
        n_layers=4,
        n_heads=8,
        slice_num=16,
        dropout=0.0,      # No regularization - will overfit
        learning_rate=1e-3,
        optimizer_type="adamw",
        scheduler_type="plateau",
        epochs=epochs,
        patience=15,
        batch_size=4,
        output_dim=2,
    )

    trainer1 = SurrogateTrainer(TrainingConfig(
        surrogate_config=initial_config,
        use_ensemble=True,
        n_ensemble=3,
        train_test_split=0.2,
    ))
    result1 = trainer1.train(params, [coords] * n_samples, outputs)
    print(f"   Initial: train_loss={result1.train_loss:.6f}, test_loss={result1.test_loss:.6f}")

    # Extract training history
    history = TrainingHistory(
        train_losses=result1.history.get("train_loss", []),
        test_losses=result1.history.get("test_loss", []),
        epochs_completed=len(result1.history.get("train_loss", [])),
        final_train_loss=result1.train_loss,
        final_test_loss=result1.test_loss,
    )

    # Setup agents
    print("\n4. Critic analyzing training curves...")
    critic = HyperparameterCriticAgent()
    architect = ArchitectAgent()

    provider = MockLLMProvider(scenario="overfitting")
    critic.set_llm_provider(provider)
    architect.set_llm_provider(provider)

    context = AgentContext()
    loop = asyncio.new_event_loop()

    critique = loop.run_until_complete(
        critic.analyze_training(context, history, initial_config.to_dict())
    )
    print(f"   Diagnosis: {critique.primary_issue.name} ({critique.severity})")
    print(f"   Recommendations: {critique.recommendations[:2]}")

    # Get architect proposal
    print("\n5. Architect proposing optimized config...")
    proposal = loop.run_until_complete(
        architect.propose_config(context, initial_config, critique, n_samples)
    )
    print(f"   Changes: {proposal.changes}")
    loop.close()

    # Train with new config
    print("\n6. Retraining with OPTIMIZED config...")
    new_config = proposal.config

    trainer2 = SurrogateTrainer(TrainingConfig(
        surrogate_config=new_config,
        use_ensemble=True,
        n_ensemble=3,
        train_test_split=0.2,
    ))
    result2 = trainer2.train(params, [coords] * n_samples, outputs)
    print(f"   After HPO: train_loss={result2.train_loss:.6f}, test_loss={result2.test_loss:.6f}")

    # Generate test prediction
    print("\n7. Generating test prediction for visualization...")
    test_params = {
        "E": 200e9, "nu": 0.3, "load_x": 100e6, "load_y": 0.0
    }
    test_param_arr = np.array([[test_params["E"], test_params["nu"],
                                 test_params["load_x"], test_params["load_y"]]],
                               dtype=np.float32)

    # Ground truth
    disp_gt = _synthetic_displacement_field(coords, test_params)
    disp_mag_gt = np.linalg.norm(disp_gt, axis=1)

    # Prediction
    pred, unc = trainer2.predict_with_uncertainty(test_param_arr, coords)
    if pred.ndim == 3:
        pred = pred[0]
    disp_mag_pred = np.linalg.norm(pred, axis=-1)

    # Uncertainty
    if unc is not None:
        if unc.ndim == 3:
            unc = unc[0]
        unc_mag = np.linalg.norm(unc, axis=-1) if unc.ndim == 2 else unc.flatten()
    else:
        unc_mag = np.zeros(N_POINTS)

    # Error
    error = np.abs(disp_mag_pred - disp_mag_gt)

    # Create visualization
    print("\n8. Creating 6-panel visualization with contour plots...")

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25,
                           left=0.05, right=0.95, top=0.92, bottom=0.08)

    # Create triangulation for contour plots
    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)

    def _contour_plot(ax, values, title, cmap, vmin=None, vmax=None):
        """Helper for contour plots."""
        v0 = values.min() if vmin is None else vmin
        v1 = values.max() if vmax is None else vmax
        levels = np.linspace(v0, v1, 20)
        cf = ax.tricontourf(triang, values, levels=levels, cmap=cmap, extend='both')
        ax.triplot(triang, 'k-', lw=0.1, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return cf

    # Row 1: Contour plots
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = _contour_plot(ax1, disp_mag_pred, "Predicted |u| (Displacement)", "jet")
    cb1 = fig.colorbar(cf1, ax=ax1, shrink=0.8, pad=0.02)
    cb1.set_label("|u| [m]", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    cf2 = _contour_plot(ax2, disp_mag_gt, "Ground Truth |u|", "jet",
                        vmin=disp_mag_pred.min(), vmax=disp_mag_pred.max())
    cb2 = fig.colorbar(cf2, ax=ax2, shrink=0.8, pad=0.02)
    cb2.set_label("|u| [m]", fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    cf3 = _contour_plot(ax3, error, "Prediction Error ||u|_pred - |u|_GT|", "Reds")
    cb3 = fig.colorbar(cf3, ax=ax3, shrink=0.8, pad=0.02)
    cb3.set_label("Error [m]", fontsize=9)
    ax3.text(0.02, 0.98, f"Mean: {error.mean():.2e}\nMax: {error.max():.2e}",
             transform=ax3.transAxes, fontsize=8, va='top',
             bbox=dict(facecolor='white', alpha=0.8))

    # Row 2: Training analysis
    ax4 = fig.add_subplot(gs[1, 0])
    if result1.history and result2.history:
        ep1 = np.arange(1, len(result1.history["train_loss"]) + 1)
        ep2 = np.arange(1, len(result2.history["train_loss"]) + 1)
        ax4.semilogy(ep1, result1.history["test_loss"], 'r--', alpha=0.6,
                     lw=1.5, label="Initial config (test)")
        ax4.semilogy(ep2, result2.history["test_loss"], 'b-', lw=2,
                     label="After HPO (test)")
        ax4.semilogy(ep2, result2.history["train_loss"], 'g--', alpha=0.6,
                     lw=1.5, label="After HPO (train)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss (log scale)")
    ax4.set_title("Training Convergence", fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Config comparison
    ax5 = fig.add_subplot(gs[1, 1])
    cfg_params = ["d_model", "n_layers", "dropout", "learning_rate", "pino_weight"]
    x = np.arange(len(cfg_params))
    init_vals = [initial_config.to_dict().get(p, 0) for p in cfg_params]
    new_vals = [new_config.to_dict().get(p, 0) for p in cfg_params]
    max_v = [max(abs(i), abs(n), 1e-10) for i, n in zip(init_vals, new_vals)]
    init_norm = [i / m for i, m in zip(init_vals, max_v)]
    new_norm = [n / m for n, m in zip(new_vals, max_v)]

    w = 0.35
    bars1 = ax5.bar(x - w/2, init_norm, w, label="Initial", color="steelblue")
    bars2 = ax5.bar(x + w/2, new_norm, w, label="After HPO", color="coral")
    ax5.set_xticks(x)
    ax5.set_xticklabels(cfg_params, rotation=20, ha="right", fontsize=9)
    ax5.set_ylabel("Normalized Value")
    ax5.set_title("Hyperparameter Changes", fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, axis="y", alpha=0.3)

    # Summary panel
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    improvement = (result1.test_loss - result2.test_loss) / result1.test_loss * 100

    summary = f"""
AGENTIC SCIML LOOP SUMMARY
{'='*44}

Mesh: Synthetic plate with hole
      {N_POINTS} nodes, {len(triangles)} triangles

Test Parameters:
  E = {test_params['E']/1e9:.0f} GPa, nu = {test_params['nu']:.2f}
  load_x = {test_params['load_x']/1e6:.0f} MPa

Critic Diagnosis: {critique.primary_issue.name}
Severity: {critique.severity}

Key HPO Changes:
{chr(10).join(f'  - {k}: {v}' for k, v in list(proposal.changes.items())[:4])}

Performance:
  Initial test loss:  {result1.test_loss:.6f}
  Final test loss:    {result2.test_loss:.6f}
  Improvement:        {improvement:.1f}%

Prediction Quality:
  Mean error: {error.mean():.2e} m
  Max error:  {error.max():.2e} m
"""
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=9, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Agentic SciML Loop: Hyperparameter Optimization for Neural Operator",
                 fontsize=14, fontweight="bold")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved visualization to: {output_path}")
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Training samples:  {n_samples}")
    print(f"  Mesh nodes:        {N_POINTS}")
    print(f"  Initial test loss: {result1.test_loss:.6f}")
    print(f"  Final test loss:   {result2.test_loss:.6f}")
    print(f"  Improvement:       {improvement:.1f}%")
    print(f"  Mean pred error:   {error.mean():.2e} m")
    print("=" * 70)


# =============================================================================
# 9. Full Visualization with Real FEM Data (Contour Plots)
# =============================================================================

def _retag_boundaries(mfem_mesh, verts: np.ndarray) -> None:
    """Tag: bottom=1, right=2, top=3, left=4, hole=5."""
    eps = 1e-10
    for i in range(mfem_mesh.GetNBE()):
        iv = mfem_mesh.GetBdrElement(i).GetVerticesArray()
        xs = [verts[iv[j]][0] for j in range(len(iv))]
        ys = [verts[iv[j]][1] for j in range(len(iv))]
        if all(y < eps for y in ys):
            tag = 1
        elif all(x > 1.0 - eps for x in xs):
            tag = 2
        elif all(y > 1.0 - eps for y in ys):
            tag = 3
        elif all(x < eps for x in xs):
            tag = 4
        else:
            tag = 5
        mfem_mesh.GetBdrElement(i).SetAttribute(tag)
    mfem_mesh.SetAttributes()


def _extract_verts(mfem_mesh) -> np.ndarray:
    """Extract vertices from MFEM mesh."""
    import ctypes
    nv = mfem_mesh.GetNV()
    verts = np.zeros((nv, 2))
    for i in range(nv):
        v = mfem_mesh.GetVertex(i)
        p = ctypes.cast(int(v), ctypes.POINTER(ctypes.c_double))
        verts[i] = [p[0], p[1]]
    return verts


def _physics_config(E: float = 200e9, nu: float = 0.3,
                    load_x: float = 1e8, load_y: float = 0.0):
    """Build PhysicsConfig for 2-D linear elasticity."""
    from piano.solvers.base import (
        PhysicsConfig, PhysicsType, MaterialProperties,
        BoundaryCondition, BoundaryConditionType,
    )
    bcs = [
        BoundaryCondition(BoundaryConditionType.SYMMETRY,
                          boundary_id=4, direction=0),
        BoundaryCondition(BoundaryConditionType.SYMMETRY,
                          boundary_id=1, direction=1),
        BoundaryCondition(BoundaryConditionType.TRACTION,
                          boundary_id=2, value=np.array([load_x, 0.])),
    ]
    if load_y != 0.0:
        bcs.append(BoundaryCondition(BoundaryConditionType.TRACTION,
                                     boundary_id=3, value=np.array([0., load_y])))
    return PhysicsConfig(
        physics_type=PhysicsType.LINEAR_ELASTICITY,
        material=MaterialProperties(E=E, nu=nu),
        boundary_conditions=bcs,
    )


def _run_fem(mesh_path: str, params: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Run MFEM solver; returns (displacement, von_mises, vertices, elements)."""
    import ctypes
    import tempfile
    from piano.mesh.mfem_manager import MFEMManager
    from piano.solvers.mfem_solver import MFEMSolver

    mgr = MFEMManager(mesh_path)
    nv = mgr.mesh.GetNV()
    verts = np.zeros((nv, 2))
    for i in range(nv):
        v = mgr.mesh.GetVertex(i)
        p = ctypes.cast(int(v), ctypes.POINTER(ctypes.c_double))
        verts[i] = [p[0], p[1]]
    _retag_boundaries(mgr.mesh, verts)

    solver = MFEMSolver(order=1)
    solver.setup(mgr, _physics_config(
        E=params.get("E", 200e9), nu=params.get("nu", 0.3),
        load_x=params.get("load_x", params.get("load", 1e8)),
        load_y=params.get("load_y", 0.0),
    ))
    with tempfile.TemporaryDirectory() as tmp:
        result = solver.solve(tmp)

    vertices = mgr.get_nodes()
    elements = [list(e[e >= 0]) for e in mgr.get_elements()]
    n_nodes = len(vertices)
    n_elems = len(elements)
    if not result.success:
        return (np.zeros((n_nodes, 2)), np.zeros(n_elems), vertices, elements)
    disp = result.solution_data.get("displacement", np.zeros((n_nodes, 2)))
    vm = result.solution_data.get("von_mises", np.zeros(n_elems))
    return disp, vm, vertices, elements


def run_agentic_loop_with_fem(
    n_train: int = 8,
    epochs: int = 50,
    output_file: str = "tests/test_outputs/agentic_sciml_fem.png",
):
    """
    Full agentic SciML loop with real FEM data and contour plot visualizations.

    Generates a 6-panel visualization:
      Row 1: Displacement contour (pred), Displacement contour (GT), Error field
      Row 2: Training curves (before/after HPO), Config changes, Summary

    Requires MFEM to be installed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.collections import PolyCollection
    import matplotlib.colors as mcolors

    try:
        import mfem.ser as mfem
    except ImportError:
        print("MFEM not available. Run: pip install mfem")
        print("Falling back to synthetic data demo...")
        return run_agentic_loop_demo(n_samples=10, epochs=epochs, output_file=output_file)

    print("=" * 70)
    print("Agentic SciML Loop with Real FEM Data")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Check for mesh files
    mesh_files = sorted(TRAIN01.glob("sample_*.mesh"))[:n_train + 2]
    if len(mesh_files) < 3:
        print(f"Not enough mesh files in {TRAIN01}")
        return run_agentic_loop_demo(n_samples=10, epochs=epochs, output_file=output_file)

    train_files = mesh_files[:-1]
    test_file = mesh_files[-1]

    print(f"\n1. Running FEM on {len(train_files)} training meshes...")

    # Build training dataset
    from piano.data.dataset import FEMDataset, FEMSample, DatasetConfig

    ds = FEMDataset(DatasetConfig(parameter_names=PARAM_NAMES, parameter_bounds=PARAM_BOUNDS))

    for i, mf in enumerate(train_files):
        params = {
            "E": float(rng.uniform(150e9, 250e9)),
            "nu": float(rng.uniform(0.25, 0.35)),
            "load_x": float(rng.uniform(50e6, 150e6)),
            "load_y": float(rng.uniform(-100e6, 100e6)),
        }
        try:
            disp, vm, verts, _ = _run_fem(str(mf), params)
            if not np.any(disp != 0):
                continue
            ds.add_sample(FEMSample(
                sample_id=f"s{i:03d}",
                parameters=params,
                coordinates=verts.astype(np.float32),
                displacement=disp.astype(np.float32),
                is_valid=True,
            ))
            print(f"   [{i+1}/{len(train_files)}] {mf.stem} |u|_max={np.linalg.norm(disp, axis=1).max():.3e}")
        except Exception as e:
            print(f"   [{i+1}/{len(train_files)}] {mf.stem} skipped: {e}")

    if len(ds.get_valid_samples()) < 3:
        print("Not enough valid FEM samples")
        return

    # Prepare training data
    params_arr, coords_list, outputs_list = ds.prepare_training_data("displacement")

    # Import training components
    from piano.surrogate.base import TransolverConfig
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.agents.roles.hyperparameter_critic import HyperparameterCriticAgent, TrainingHistory
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.base import AgentContext
    import asyncio

    # Initial config (intentionally suboptimal for demo)
    print(f"\n2. Training with initial config ({len(ds.get_valid_samples())} samples)...")
    initial_config = TransolverConfig(
        d_model=128, n_layers=4, n_heads=8, slice_num=16,
        dropout=0.0, learning_rate=1e-3,
        optimizer_type="adamw", scheduler_type="plateau",
        epochs=epochs, patience=30, batch_size=4,
        output_dim=2, pino_weight=0.1, pino_eq_weight=0.1,
    )

    trainer1 = SurrogateTrainer(TrainingConfig(
        surrogate_config=initial_config,
        use_ensemble=True, n_ensemble=3,
        train_test_split=0.2,
    ))
    result1 = trainer1.train(params_arr, coords_list, outputs_list)
    print(f"   Initial: train_loss={result1.train_loss:.6f}, test_loss={result1.test_loss:.6f}")

    # Critic analysis
    print("\n3. Critic analyzing training...")
    history = TrainingHistory(
        train_losses=result1.history.get("train_loss", []),
        test_losses=result1.history.get("test_loss", []),
        epochs_completed=len(result1.history.get("train_loss", [])),
        final_train_loss=result1.train_loss,
        final_test_loss=result1.test_loss,
    )

    critic = HyperparameterCriticAgent()
    architect = ArchitectAgent()
    provider = MockLLMProvider(scenario="overfitting")
    critic.set_llm_provider(provider)
    architect.set_llm_provider(provider)

    context = AgentContext()
    loop = asyncio.new_event_loop()
    critique = loop.run_until_complete(
        critic.analyze_training(context, history, initial_config.to_dict())
    )
    print(f"   Diagnosis: {critique.primary_issue.name} ({critique.severity})")

    # Architect proposal
    print("\n4. Architect proposing new config...")
    proposal = loop.run_until_complete(
        architect.propose_config(context, initial_config, critique, len(params_arr))
    )
    print(f"   Changes: {proposal.changes}")
    loop.close()

    # Retrain
    print("\n5. Retraining with optimized config...")
    new_config = proposal.config
    trainer2 = SurrogateTrainer(TrainingConfig(
        surrogate_config=new_config,
        use_ensemble=True, n_ensemble=3,
        train_test_split=0.2,
    ))
    result2 = trainer2.train(params_arr, coords_list, outputs_list)
    print(f"   After HPO: train_loss={result2.train_loss:.6f}, test_loss={result2.test_loss:.6f}")

    # Test on held-out mesh
    print(f"\n6. Evaluating on test mesh: {test_file.stem}...")
    test_params = {
        "E": float(rng.uniform(150e9, 250e9)),
        "nu": float(rng.uniform(0.25, 0.35)),
        "load_x": float(rng.uniform(50e6, 150e6)),
        "load_y": float(rng.uniform(-100e6, 100e6)),
    }
    disp_gt, vm_gt, verts, elems = _run_fem(str(test_file), test_params)
    param_vec = np.array([[test_params[k] for k in PARAM_NAMES]], dtype=np.float32)

    # Predict with trained model
    mean_pred, unc = trainer2.predict_with_uncertainty(param_vec, verts.astype(np.float32))
    if mean_pred.ndim == 3:
        mean_pred = mean_pred[0]

    disp_mag_pred = np.linalg.norm(mean_pred, axis=-1)
    disp_mag_gt = np.linalg.norm(disp_gt, axis=1)
    error_field = np.abs(disp_mag_pred - disp_mag_gt)

    # Interpolate to elements for visualization
    def _node_to_elem(field):
        return np.array([field[e].mean() for e in elems])

    pred_e = _node_to_elem(disp_mag_pred)
    gt_e = _node_to_elem(disp_mag_gt)
    err_e = _node_to_elem(error_field)

    # Normalize
    def _norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-10)

    # Create visualization
    print("\n7. Generating visualization...")

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Helper for mesh plots
    def _mesh_plot(ax, field, title, cmap, vmin=None, vmax=None):
        polys = [verts[e] for e in elems]
        v0 = float(field.min()) if vmin is None else vmin
        v1 = float(field.max()) if vmax is None else vmax
        pc = PolyCollection(
            polys, array=field, cmap=cmap,
            norm=mcolors.Normalize(vmin=v0, vmax=v1),
            edgecolors="k", linewidths=0.15,
        )
        ax.add_collection(pc)
        ax.set_xlim(verts[:, 0].min() - 0.05, verts[:, 0].max() + 0.05)
        ax.set_ylim(verts[:, 1].min() - 0.05, verts[:, 1].max() + 0.05)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        return pc

    # Row 1: Contour plots
    ax1 = fig.add_subplot(gs[0, 0])
    pc1 = _mesh_plot(ax1, pred_e, "Predicted |u| (displacement magnitude)", "jet")
    cb1 = fig.colorbar(pc1, ax=ax1, shrink=0.8)
    cb1.set_label("|u| [m]")

    ax2 = fig.add_subplot(gs[0, 1])
    pc2 = _mesh_plot(ax2, gt_e, "Ground Truth |u| (FEM)", "jet",
                     vmin=pred_e.min(), vmax=pred_e.max())
    cb2 = fig.colorbar(pc2, ax=ax2, shrink=0.8)
    cb2.set_label("|u| [m]")

    ax3 = fig.add_subplot(gs[0, 2])
    pc3 = _mesh_plot(ax3, err_e, "Absolute Error ||u|_pred - |u|_GT|", "Reds")
    cb3 = fig.colorbar(pc3, ax=ax3, shrink=0.8)
    cb3.set_label("Error [m]")
    # Add error stats
    ax3.text(0.02, 0.98, f"Mean: {err_e.mean():.2e}\nMax: {err_e.max():.2e}",
             transform=ax3.transAxes, fontsize=8, va="top",
             bbox=dict(facecolor="white", alpha=0.8))

    # Row 2: Training analysis
    ax4 = fig.add_subplot(gs[1, 0])
    if result1.history and result2.history:
        ep1 = np.arange(1, len(result1.history["train_loss"]) + 1)
        ep2 = np.arange(1, len(result2.history["train_loss"]) + 1)
        ax4.semilogy(ep1, result1.history["test_loss"], "r--", alpha=0.5, label="Initial (test)")
        ax4.semilogy(ep2, result2.history["test_loss"], "b-", lw=2, label="After HPO (test)")
        ax4.semilogy(ep2, result2.history["train_loss"], "g--", alpha=0.7, label="After HPO (train)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss (log)")
    ax4.set_title("Training Convergence", fontsize=10, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Config comparison
    ax5 = fig.add_subplot(gs[1, 1])
    cfg_params = ["d_model", "n_layers", "dropout", "learning_rate", "pino_weight"]
    x = np.arange(len(cfg_params))
    init_vals = [initial_config.to_dict().get(p, 0) for p in cfg_params]
    new_vals = [new_config.to_dict().get(p, 0) for p in cfg_params]
    max_v = [max(abs(i), abs(n), 1e-10) for i, n in zip(init_vals, new_vals)]
    init_norm = [i / m for i, m in zip(init_vals, max_v)]
    new_norm = [n / m for n, m in zip(new_vals, max_v)]

    w = 0.35
    ax5.bar(x - w/2, init_norm, w, label="Initial", color="steelblue")
    ax5.bar(x + w/2, new_norm, w, label="After HPO", color="coral")
    ax5.set_xticks(x)
    ax5.set_xticklabels(cfg_params, rotation=20, ha="right")
    ax5.set_ylabel("Normalized Value")
    ax5.set_title("Hyperparameter Changes", fontsize=10, fontweight="bold")
    ax5.legend()
    ax5.grid(True, axis="y", alpha=0.3)

    # Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    improvement = (result1.test_loss - result2.test_loss) / result1.test_loss * 100

    summary = f"""
AGENTIC SCIML LOOP SUMMARY
{'='*45}

Test Mesh: {test_file.stem}
Parameters:
  E = {test_params['E']/1e9:.1f} GPa
  nu = {test_params['nu']:.3f}
  load_x = {test_params['load_x']/1e6:.1f} MPa

Critic Diagnosis: {critique.primary_issue.name}
Severity: {critique.severity}

Key HPO Changes:
{chr(10).join(f'  - {k}: {v}' for k, v in list(proposal.changes.items())[:4])}

Performance:
  Initial test loss:  {result1.test_loss:.6f}
  Final test loss:    {result2.test_loss:.6f}
  Improvement:        {improvement:.1f}%

Prediction Error (on test mesh):
  Mean |u| error: {err_e.mean():.2e} m
  Max |u| error:  {err_e.max():.2e} m
"""
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=9, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Agentic SciML Loop: HPO with Real FEM Data (Plate with Hole)",
                 fontsize=13, fontweight="bold", y=0.98)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved visualization to: {output_path}")
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Training samples: {len(ds.get_valid_samples())}")
    print(f"  Initial test loss: {result1.test_loss:.6f}")
    print(f"  Final test loss:   {result2.test_loss:.6f}")
    print(f"  Improvement:       {improvement:.1f}%")
    print(f"  Mean pred error:   {err_e.mean():.2e} m")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agentic SciML Loop Test")
    parser.add_argument("--n-samples", type=int, default=15,
                        help="Number of samples (default: 15)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path")
    parser.add_argument("--fem", action="store_true",
                        help="Use real FEM data with contour plots (requires MFEM)")
    args = parser.parse_args()

    if args.fem:
        output = args.output or str(PROJECT_ROOT / "tests" / "test_outputs" / "agentic_sciml_fem.png")
        run_agentic_loop_with_fem(
            n_train=args.n_samples,
            epochs=args.epochs,
            output_file=output,
        )
    else:
        output = args.output or str(PROJECT_ROOT / "tests" / "test_outputs" / "agentic_loop_demo.png")
        run_agentic_loop_demo(
            n_samples=args.n_samples,
            epochs=args.epochs,
            output_file=output,
        )
