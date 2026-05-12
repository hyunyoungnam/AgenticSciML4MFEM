"""
tests/test_agentic_sciml.py — Test the Agentic SciML Loop for Crack Problems.

Tests the complete agentic hyperparameter optimization pipeline:
  1. HyperparameterCriticAgent - diagnoses training issues
  2. ArchitectAgent - proposes architecture/optimizer changes
  3. PhysicistAgent - proposes physics loss configuration changes
  4. AgenticSurrogateTrainer - full training loop with 3-agent HPO

Focus: Static crack problems with stress singularity at crack tip.
The singularity (1/sqrt(r)) is challenging for neural operators and
benefits from agentic HPO to tune architecture and physics constraints.

Run:
    pytest tests/test_agentic_sciml.py -v
    pytest tests/test_agentic_sciml.py -v -m "not mfem"
    python tests/test_agentic_sciml.py [--n-samples N] [--epochs E]
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
CRACK_DATA = PROJECT_ROOT / "crack_data"

# Crack problem parameters
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "E": (150e9, 250e9),           # Young's modulus [Pa]
    "nu": (0.25, 0.35),            # Poisson's ratio
    "K_I": (1e6, 10e6),            # Mode I stress intensity factor [Pa*sqrt(m)]
    "crack_length": (0.2, 0.5),    # Crack length ratio a/W
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

    def __init__(self, scenario: str = "underfitting"):
        """
        Initialize with a predefined scenario.
        For crack problems, underfitting is common due to singularity.
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

        if "training analyst" in system_prompt.lower():
            return MockLLMResponse(self._critic_response())
        elif "architect" in system_prompt.lower():
            return MockLLMResponse(self._architect_response())
        elif "physicist" in system_prompt.lower():
            return MockLLMResponse(self._physicist_response())
        elif "scientific machine learning" in system_prompt.lower():
            return MockLLMResponse(self._proposer_response())
        else:
            return MockLLMResponse("Unknown agent type")

    def _critic_response(self) -> str:
        """Generate mock critic response."""
        responses = {
            "underfitting": """
DIAGNOSIS: The model is underfitting, particularly near the crack tip where stress singularity occurs. Both training and test losses remain high. The neural operator lacks capacity to capture the 1/sqrt(r) singularity behavior.

PRIMARY_ISSUE: UNDERFITTING
SEVERITY: high

RECOMMENDATIONS:
- Increase d_model from 64 to 256 for more representational capacity
- Increase n_layers from 2 to 6 for deeper feature extraction
- Use SiLU activation instead of GELU (better for sharp gradients)
- Increase slice_num for finer spatial resolution near crack tip
- Reduce PINO weight initially to allow data-driven learning of singularity

SHOULD_RETRAIN: true

METRICS_ANALYSIS:
- train_test_gap: Small but both losses are high
- convergence_rate: Plateaued at suboptimal level
- crack_tip_error: Significantly higher than bulk error
""",
            "overfitting": """
DIAGNOSIS: Model is overfitting to training samples. Training loss is low but test loss diverges. The model memorizes specific crack configurations instead of learning general fracture mechanics.

PRIMARY_ISSUE: OVERFITTING
SEVERITY: high

RECOMMENDATIONS:
- Increase dropout from 0.0 to 0.2
- Reduce d_model from 256 to 128
- Increase PINO weight to enforce physics constraints
- Add weight decay to optimizer

SHOULD_RETRAIN: true
""",
            "slow_convergence": """
DIAGNOSIS: Training converges too slowly. Learning rate is too conservative for this problem.

PRIMARY_ISSUE: SLOW_CONVERGENCE
SEVERITY: medium

RECOMMENDATIONS:
- Increase learning rate from 1e-4 to 1e-3
- Use cosine scheduler with warm restarts
- Consider AdamW optimizer

SHOULD_RETRAIN: true
""",
            "stable": """
DIAGNOSIS: Training appears healthy. Model is learning the crack tip singularity reasonably well.

PRIMARY_ISSUE: NONE
SEVERITY: low

SHOULD_RETRAIN: false
""",
        }
        return responses.get(self.scenario, responses["underfitting"])

    def _architect_response(self) -> str:
        """Generate mock architect response."""
        # Conservative changes for small datasets (10-20 samples)
        # Avoid overfitting by not increasing model too aggressively
        responses = {
            "underfitting": """
REASONING: For small datasets with singularity, we need moderate capacity increase with strong regularization. Aggressive capacity increase will cause overfitting. Focus on learning rate and activation.

CHANGES:
- d_model: 64 (keep moderate for small dataset)
- n_layers: 3 (slight increase)
- n_heads: 4 (keep moderate)
- slice_num: 16 (moderate spatial resolution)
- activation: silu (better for sharp gradients)
- learning_rate: 2e-4 (moderate increase)
- dropout: 0.1 (regularization for small dataset)
- pino_weight: 0.1 (physics constraints help generalization)

EXPECTED_IMPACT: Better generalization on small dataset while capturing singularity.

CONFIDENCE: medium
""",
            "overfitting": """
REASONING: Need to regularize and reduce capacity to prevent memorization.

CHANGES:
- d_model: 48 (reduced)
- dropout: 0.15 (added regularization)
- pino_weight: 0.15 (physics constraints)
- learning_rate: 1e-4 (reduced)

CONFIDENCE: high
""",
            "slow_convergence": """
REASONING: Need faster learning dynamics.

CHANGES:
- learning_rate: 3e-4 (moderate increase)
- scheduler_type: cosine
- optimizer_type: adamw

CONFIDENCE: medium
""",
            "stable": """
REASONING: Minor tuning only.

CHANGES:
- epochs: 50

CONFIDENCE: medium
""",
        }
        return responses.get(self.scenario, responses["underfitting"])

    def _physicist_response(self) -> str:
        """Generate mock physicist response for physics loss configuration."""
        responses = {
            "underfitting": """
PHYSICS_DIAGNOSIS: For small datasets, physics loss provides crucial regularization. Moderate physics weights help the model generalize by enforcing physical constraints. Don't increase too aggressively to avoid conflicting gradients.

CHANGES:
- pino_weight: 0.1 (moderate increase for regularization)
- pino_eq_weight: 0.1 (balanced with data loss)

REASONING: Physics constraints act as regularizers on small datasets. Moderate weights help without overwhelming the data-driven learning.

EXPECTED_IMPACT: Better generalization through physics-based regularization.

CONFIDENCE: medium
""",
            "overfitting": """
PHYSICS_DIAGNOSIS: Physics constraints may be too weak, allowing model to memorize training data without learning underlying physics. Increasing physics loss will regularize the model.

CHANGES:
- pino_weight: 0.3 (significantly increased)
- pino_eq_weight: 0.25 (increased for physics regularization)

REASONING: Physics-informed loss acts as a regularizer by enforcing physical constraints that generalize across samples.

EXPECTED_IMPACT: Reduced overfitting through physics-based regularization.

CONFIDENCE: medium
""",
            "slow_convergence": """
PHYSICS_DIAGNOSIS: Physics loss may be conflicting with data loss early in training. Reduce physics weights initially.

CHANGES:
- pino_weight: 0.05 (reduced for faster initial convergence)
- pino_eq_weight: 0.05 (reduced)

REASONING: Let the model learn from data first, then physics constraints will refine the solution.

EXPECTED_IMPACT: Faster initial convergence.

CONFIDENCE: medium
""",
            "stable": """
PHYSICS_DIAGNOSIS: Physics loss configuration appears appropriate. Minor tuning may help.

CHANGES:
- pino_eq_weight: 0.12 (slight increase for better equilibrium)

REASONING: Fine-tuning for optimal balance.

EXPECTED_IMPACT: Marginal improvement in physics consistency.

CONFIDENCE: low
""",
        }
        return responses.get(self.scenario, responses["underfitting"])

    def _proposer_response(self) -> str:
        """Generate mock adaptive proposer response."""
        # Generate proposals based on crack problem parameters
        return """
**Proposal 1**
Parameters: E=200e9, nu=0.30, K_I=6e6, crack_length=0.35
Target Region: High uncertainty near crack tip with long crack
Reasoning: The model shows highest uncertainty in the region with crack_length > 0.3 and high K_I values. This sample targets the singularity-dominated regime where the 1/sqrt(r) behavior is most pronounced.
Expected Improvement: Should reduce error in high-stress intensity region by ~15%
Priority: High

**Proposal 2**
Parameters: E=180e9, nu=0.28, K_I=4e6, crack_length=0.25
Target Region: Moderate crack length with lower stiffness
Reasoning: Current dataset lacks samples with lower Young's modulus combined with moderate crack lengths. This configuration will help the model generalize across material stiffness variations.
Expected Improvement: Better generalization for softer materials
Priority: Medium

**Proposal 3**
Parameters: E=220e9, nu=0.32, K_I=8e6, crack_length=0.45
Target Region: Long crack with high stress intensity
Reasoning: Extreme case near parameter bounds. The model needs exposure to near-critical crack configurations to accurately predict failure-prone scenarios.
Expected Improvement: Improved accuracy at parameter space boundaries
Priority: High

**Proposal 4**
Parameters: E=190e9, nu=0.30, K_I=3e6, crack_length=0.30
Target Region: Central parameter space with low K_I
Reasoning: Filling coverage gap in the low stress intensity region with moderate crack length.
Expected Improvement: More uniform error distribution across parameter space
Priority: Medium

**Proposal 5**
Parameters: E=210e9, nu=0.33, K_I=7e6, crack_length=0.40
Target Region: High Poisson's ratio with long crack
Reasoning: Higher Poisson's ratio affects stress distribution near crack tip. This sample explores material incompressibility effects on fracture behavior.
Expected Improvement: Better capture of Poisson's ratio sensitivity
Priority: Medium
"""


# =============================================================================
# Crack-Specific Synthetic Data Generation
# =============================================================================

def _generate_crack_mesh(n_points: int = 500, crack_length: float = 0.3,
                          crack_angle: float = 0.0, rng: np.random.Generator = None):
    """
    Generate synthetic mesh for plate with edge crack.
    Returns coordinates and triangulation.
    """
    from scipy.spatial import Delaunay

    if rng is None:
        rng = np.random.default_rng(42)

    # Crack tip position
    angle_rad = np.radians(crack_angle)
    tip_x = crack_length * np.cos(angle_rad)
    tip_y = 0.5 + crack_length * np.sin(angle_rad)

    points = []

    # Boundary points
    n_boundary = 40
    for i in range(n_boundary):
        t = i / n_boundary
        points.append([t, 0.0])
        points.append([1.0, t])
        points.append([1.0 - t, 1.0])
        points.append([0.0, 1.0 - t])

    # Interior points avoiding crack
    while len(points) < n_points:
        p = rng.uniform(0, 1, size=(2,))
        # Check if point is on crack line (from origin to tip)
        if p[0] < tip_x:
            # Distance from crack line
            if abs(p[1] - 0.5) > 0.02:  # Not too close to crack
                points.append(p.tolist())
        else:
            points.append(p.tolist())

    # Refined points near crack tip
    for level in range(4):
        r = 0.1 * (0.5 ** level)
        n_ring = 12 * (level + 1)
        for i in range(n_ring):
            theta = 2 * np.pi * i / n_ring
            x = tip_x + r * np.cos(theta)
            y = tip_y + r * np.sin(theta)
            if 0 < x < 1 and 0 < y < 1:
                points.append([x, y])

    # Points along crack faces
    for i in range(1, 20):
        t = i / 20 * crack_length
        x = t * np.cos(angle_rad)
        y_base = 0.5 + t * np.sin(angle_rad)
        points.append([x, y_base + 0.01])  # Upper face
        points.append([x, y_base - 0.01])  # Lower face

    coords = np.array(points, dtype=np.float32)

    # Triangulate
    tri = Delaunay(coords)

    # Filter triangles crossing crack
    valid = []
    for simplex in tri.simplices:
        centroid = coords[simplex].mean(axis=0)
        # Skip if centroid is on crack
        if centroid[0] < tip_x and abs(centroid[1] - 0.5) < 0.015:
            continue
        valid.append(simplex)

    return coords, np.array(valid)


def _williams_displacement(coords: np.ndarray, params: Dict,
                            tip_x: float, tip_y: float = 0.5) -> np.ndarray:
    """
    Generate Williams expansion displacement field near crack tip.
    This is the analytical solution for mode I crack tip displacement.

    u_x = K_I / (2*mu) * sqrt(r/(2*pi)) * cos(theta/2) * (kappa - 1 + 2*sin^2(theta/2))
    u_y = K_I / (2*mu) * sqrt(r/(2*pi)) * sin(theta/2) * (kappa + 1 - 2*cos^2(theta/2))

    where kappa = (3 - nu)/(1 + nu) for plane stress
    """
    E = params.get("E", 200e9)
    nu = params.get("nu", 0.3)
    K_I = params.get("K_I", 5e6)  # Stress intensity factor

    mu = E / (2 * (1 + nu))  # Shear modulus
    kappa = (3 - nu) / (1 + nu)  # Plane stress

    n_points = len(coords)
    disp = np.zeros((n_points, 2), dtype=np.float32)

    for i, (x, y) in enumerate(coords):
        # Distance and angle from crack tip
        dx = x - tip_x
        dy = y - tip_y
        r = np.sqrt(dx**2 + dy**2) + 1e-10  # Avoid division by zero
        theta = np.arctan2(dy, dx)

        # Williams expansion (mode I)
        sqrt_r = np.sqrt(r / (2 * np.pi))
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)

        # Displacement components
        ux = K_I / (2 * mu) * sqrt_r * cos_half * (kappa - 1 + 2 * sin_half**2)
        uy = K_I / (2 * mu) * sqrt_r * sin_half * (kappa + 1 - 2 * cos_half**2)

        disp[i, 0] = ux
        disp[i, 1] = uy

    return disp


# =============================================================================
# 1. HyperparameterCriticAgent Tests
# =============================================================================

def test_critic_detect_issues_heuristic_overfitting():
    """Test heuristic overfitting detection."""
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory, TrainingIssue,
    )

    critic = HyperparameterCriticAgent()

    # Smooth overfitting curve
    train_losses = [0.5 - 0.02 * i for i in range(25)]
    test_losses = [0.5 - 0.01 * i if i < 10 else 0.4 + 0.02 * (i - 10) for i in range(25)]

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
    """Test heuristic underfitting detection."""
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory, TrainingIssue,
    )

    critic = HyperparameterCriticAgent()

    # Both losses high and plateaued
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
        HyperparameterCriticAgent, TrainingHistory, TrainingIssue,
    )

    critic = HyperparameterCriticAgent()

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
        HyperparameterCriticAgent, TrainingHistory, TrainingIssue,
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
        HyperparameterCriticAgent, TrainingHistory,
    )

    critic = HyperparameterCriticAgent()

    # Good training - should NOT trigger
    good_history = TrainingHistory(
        train_losses=[0.1, 0.05, 0.02, 0.01, 0.005],
        test_losses=[0.12, 0.06, 0.03, 0.015, 0.008],
        epochs_completed=5,
        final_test_loss=0.008,
    )
    assert not critic.should_trigger_hpo(good_history, threshold=0.01)

    # Bad training - SHOULD trigger
    bad_history = TrainingHistory(
        train_losses=[0.5, 0.4, 0.3, 0.2, 0.1],
        test_losses=[0.5, 0.5, 0.55, 0.6, 0.7],
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
        HyperparameterCriticAgent, TrainingHistory, TrainingIssue,
    )

    critic = HyperparameterCriticAgent()
    provider = MockLLMProvider(scenario="underfitting")
    critic.set_llm_provider(provider)

    context = AgentContext()
    history = TrainingHistory(
        train_losses=[0.5, 0.45, 0.42, 0.40],
        test_losses=[0.55, 0.50, 0.48, 0.47],
        epochs_completed=4,
        final_train_loss=0.40,
        final_test_loss=0.47,
    )

    result = await critic.analyze_training(
        context=context,
        training_history=history,
        config={"d_model": 64, "n_layers": 2},
    )

    assert result.primary_issue == TrainingIssue.UNDERFITTING
    assert result.severity == "high"
    assert result.should_retrain is True
    assert provider.call_count == 1


# =============================================================================
# 2. ArchitectAgent Tests
# =============================================================================

@pytest.mark.asyncio
async def test_architect_propose_config_underfitting():
    """Test architect proposes config for underfitting (crack tip singularity)."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.roles.hyperparameter_critic import CritiqueResult, TrainingIssue
    from piano.surrogate.base import TransolverConfig

    architect = ArchitectAgent()
    provider = MockLLMProvider(scenario="underfitting")
    architect.set_llm_provider(provider)

    context = AgentContext()
    current_config = TransolverConfig(
        d_model=64, n_layers=2, dropout=0.0, learning_rate=1e-4
    )
    critique = CritiqueResult(
        primary_issue=TrainingIssue.UNDERFITTING,
        severity="high",
        diagnosis="Model cannot capture crack tip singularity",
        recommendations=["Increase capacity", "Use SiLU activation"],
        should_retrain=True,
    )

    proposal = await architect.propose_config(
        context=context,
        current_config=current_config,
        critique=critique,
        dataset_size=10,
    )

    assert proposal.config is not None
    assert proposal.changes.get("d_model", 64) >= 64
    assert len(proposal.reasoning) > 0


def test_architect_apply_changes():
    """Test architect applies changes correctly."""
    from piano.agents.roles.architect import ArchitectAgent
    from piano.surrogate.base import TransolverConfig

    architect = ArchitectAgent()

    base_config = TransolverConfig(
        d_model=64, n_layers=2, dropout=0.0, learning_rate=1e-4,
    )

    changes = {
        "d_model": 256,
        "n_layers": 6,
        "dropout": 0.05,
        "learning_rate": 1e-3,
    }

    new_config = architect.apply_changes(base_config, changes)

    assert new_config.d_model == 256
    assert new_config.n_layers == 6
    assert new_config.dropout == 0.05
    assert new_config.learning_rate == 1e-3


# =============================================================================
# 2b. Physicist Agent Tests
# =============================================================================

@pytest.mark.asyncio
async def test_physicist_propose_physics_config():
    """Test physicist proposes physics loss configuration."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.physicist import PhysicistAgent
    from piano.agents.roles.hyperparameter_critic import CritiqueResult, TrainingIssue

    physicist = PhysicistAgent()
    provider = MockLLMProvider(scenario="underfitting")
    physicist.set_llm_provider(provider)

    context = AgentContext()
    current_config = {
        "pino_weight": 0.1,
        "pino_eq_weight": 0.1,
        "pino_E": 1.0,
        "pino_nu": 0.3,
    }
    critique = CritiqueResult(
        primary_issue=TrainingIssue.UNDERFITTING,
        severity="high",
        diagnosis="Model cannot capture crack tip singularity",
        recommendations=["Increase physics enforcement"],
        should_retrain=True,
    )

    proposal = await physicist.propose_physics_config(
        context=context,
        current_config=current_config,
        critique=critique,
        dataset_size=10,
        problem_type="crack",
        has_singularity=True,
    )

    assert proposal.changes is not None
    assert len(proposal.physics_diagnosis) > 0
    # Should propose increased physics weights for underfitting
    if "pino_weight" in proposal.changes:
        assert proposal.changes["pino_weight"] >= 0.1


def test_physicist_detect_physics_issues():
    """Test physicist heuristic issue detection."""
    from piano.agents.roles.physicist import PhysicistAgent, PhysicsIssue
    from piano.agents.roles.hyperparameter_critic import TrainingHistory

    physicist = PhysicistAgent()

    # Case 1: PINO loss not decreasing (weak enforcement)
    history = TrainingHistory(
        train_losses=[0.5, 0.3, 0.2, 0.15, 0.1],
        test_losses=[0.6, 0.4, 0.3, 0.25, 0.2],
        pino_losses=[0.5, 0.48, 0.47, 0.46, 0.46],  # Barely decreasing
        epochs_completed=5,
    )
    issues = physicist.detect_physics_issues(history, {"pino_weight": 0.1})
    assert PhysicsIssue.WEAK_PHYSICS_ENFORCEMENT in issues

    # Case 2: Overly strong physics (high pino_weight, train loss stuck)
    history2 = TrainingHistory(
        train_losses=[0.5, 0.48, 0.47, 0.46, 0.45],  # Barely decreasing
        test_losses=[0.6, 0.58, 0.57, 0.56, 0.55],
        pino_losses=[0.1, 0.08, 0.06, 0.04, 0.02],  # PINO decreasing
        epochs_completed=5,
    )
    issues2 = physicist.detect_physics_issues(history2, {"pino_weight": 0.6})
    assert PhysicsIssue.OVERLY_STRONG_PHYSICS in issues2


def test_physicist_should_consult():
    """Test physicist consultation trigger logic."""
    from piano.agents.roles.physicist import PhysicistAgent
    from piano.agents.roles.hyperparameter_critic import TrainingHistory

    physicist = PhysicistAgent()

    # Should consult when physics issues detected (PINO loss not decreasing)
    history_bad = TrainingHistory(
        train_losses=[0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12],
        test_losses=[0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.25, 0.22],
        pino_losses=[0.5, 0.49, 0.48, 0.48, 0.47, 0.47, 0.46, 0.46, 0.46, 0.46],  # Barely decreasing
        epochs_completed=10,
    )
    assert physicist.should_consult(history_bad, {"pino_weight": 0.1})

    # Should not consult when no physics issues (PINO decreasing well)
    history_good = TrainingHistory(
        train_losses=[0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12],
        test_losses=[0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.25, 0.22],
        pino_losses=[0.5, 0.4, 0.32, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04],  # Decreasing well
        epochs_completed=10,
    )
    assert not physicist.should_consult(history_good, {"pino_weight": 0.1})


# =============================================================================
# 3. TransolverConfig Tests
# =============================================================================

def test_transolver_config_all_tunable_params():
    """Test that TransolverConfig includes all tunable parameters."""
    from piano.surrogate.base import TransolverConfig

    config = TransolverConfig(
        d_model=256, n_layers=6, n_heads=8, slice_num=32,
        mlp_ratio=4.0, dropout=0.1, activation="silu",
        optimizer_type="adamw", learning_rate=1e-3, scheduler_type="cosine",
        pino_weight=0.1, pino_eq_weight=0.1,
        batch_size=32, epochs=100, patience=50,
    )

    d = config.to_dict()
    assert d["d_model"] == 256
    assert d["activation"] == "silu"


# =============================================================================
# 4. Agentic Training Tests (Synthetic Crack Data)
# =============================================================================

@pytest.fixture(scope="module")
def synthetic_crack_dataset():
    """Create synthetic crack dataset for testing."""
    rng = np.random.default_rng(42)
    N_SAMPLES = 10

    # Generate crack mesh
    coords, triangles = _generate_crack_mesh(n_points=400, crack_length=0.3, rng=rng)

    # Generate parameter samples
    params = []
    outputs = []

    for i in range(N_SAMPLES):
        p = {
            "E": float(rng.uniform(150e9, 250e9)),
            "nu": float(rng.uniform(0.25, 0.35)),
            "K_I": float(rng.uniform(1e6, 10e6)),
            "crack_length": 0.3,
        }
        params.append([p["E"], p["nu"], p["K_I"], p["crack_length"]])

        # Williams expansion displacement
        disp = _williams_displacement(coords, p, tip_x=0.3, tip_y=0.5)
        outputs.append(disp)

    params_arr = np.array(params, dtype=np.float32)

    return params_arr, coords, triangles, outputs


def test_agentic_trainer_initialization():
    """Test AgenticSurrogateTrainer initialization."""
    from piano.surrogate.agentic_trainer import (
        AgenticSurrogateTrainer, AgenticTrainingConfig,
    )
    from piano.surrogate.base import TransolverConfig

    config = AgenticTrainingConfig(
        base_config=TransolverConfig(d_model=64, n_layers=2),
        max_hpo_rounds=3,
        trigger_threshold=0.1,
    )

    provider = MockLLMProvider()
    trainer = AgenticSurrogateTrainer(config, llm_provider=provider)

    assert trainer.config == config
    assert trainer.critic is not None
    assert trainer.architect is not None


def test_agentic_trainer_train_without_hpo(synthetic_crack_dataset):
    """Test agentic training when HPO is not needed."""
    from piano.surrogate.agentic_trainer import (
        AgenticSurrogateTrainer, AgenticTrainingConfig,
    )
    from piano.surrogate.base import TransolverConfig

    params, coords, _, outputs = synthetic_crack_dataset

    config = AgenticTrainingConfig(
        base_config=TransolverConfig(
            d_model=32, n_layers=1, n_heads=2, slice_num=4,
            epochs=5, patience=10, batch_size=4, output_dim=2,
        ),
        max_hpo_rounds=2,
        trigger_threshold=100.0,  # High = no HPO
        use_ensemble=False,
    )

    provider = MockLLMProvider(scenario="stable")
    trainer = AgenticSurrogateTrainer(config, llm_provider=provider)

    result = trainer.train(params, [coords] * len(params), outputs)

    assert result.success
    assert result.n_hpo_rounds == 0


# =============================================================================
# 5. Critic-Architect Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_critic_architect_loop_crack():
    """Test critic-architect loop for crack problem."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory,
    )
    from piano.agents.roles.architect import ArchitectAgent
    from piano.surrogate.base import TransolverConfig

    critic = HyperparameterCriticAgent()
    architect = ArchitectAgent()

    provider = MockLLMProvider(scenario="underfitting")
    critic.set_llm_provider(provider)
    architect.set_llm_provider(provider)

    context = AgentContext()

    # Small model struggling with singularity
    current_config = TransolverConfig(
        d_model=64, n_layers=2, dropout=0.0, learning_rate=1e-4,
    )

    # High loss plateau (underfitting)
    history = TrainingHistory(
        train_losses=[0.5, 0.45, 0.42, 0.40, 0.39],
        test_losses=[0.55, 0.50, 0.47, 0.45, 0.44],
        epochs_completed=5,
        final_train_loss=0.39,
        final_test_loss=0.44,
    )

    # Critic analyzes
    critique = await critic.analyze_training(
        context=context,
        training_history=history,
        config=current_config.to_dict(),
    )

    assert critique.should_retrain is True
    assert critique.primary_issue.name == "UNDERFITTING"

    # Architect proposes fix
    proposal = await architect.propose_config(
        context=context,
        current_config=current_config,
        critique=critique,
        dataset_size=10,
    )

    assert proposal.config is not None
    # Should increase capacity for crack singularity
    assert proposal.changes.get("d_model", 64) >= 64


# =============================================================================
# 6. Parametric Scenario Tests
# =============================================================================

@pytest.mark.parametrize("scenario,expected_issue", [
    ("underfitting", "UNDERFITTING"),
    ("overfitting", "OVERFITTING"),
    ("slow_convergence", "SLOW_CONVERGENCE"),
    ("stable", "NONE"),
])
@pytest.mark.asyncio
async def test_critic_scenarios(scenario, expected_issue):
    """Test critic identifies different scenarios."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory,
    )

    critic = HyperparameterCriticAgent()
    provider = MockLLMProvider(scenario=scenario)
    critic.set_llm_provider(provider)

    context = AgentContext()
    history = TrainingHistory(
        train_losses=[0.5, 0.4, 0.3, 0.2, 0.1],
        test_losses=[0.5, 0.45, 0.4, 0.35, 0.3],
        epochs_completed=5,
    )

    result = await critic.analyze_training(
        context=context,
        training_history=history,
        config={"d_model": 128},
    )

    assert result.primary_issue.name == expected_issue


# =============================================================================
# 7. AdaptiveProposer Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_adaptive_proposer_propose_targeted():
    """Test AdaptiveProposerAgent produces targeted proposals."""
    from piano.agents.roles.adaptive_proposer import AdaptiveProposerAgent
    from piano.agents.base import AgentContext
    from piano.surrogate.evaluator import UncertaintyAnalysis, WeakRegion

    proposer = AdaptiveProposerAgent()
    provider = MockLLMProvider(scenario="underfitting")
    proposer.set_llm_provider(provider)

    context = AgentContext()

    # Create mock uncertainty analysis with correct fields
    uncertainty = UncertaintyAnalysis(
        overall_uncertainty=0.15,
        max_uncertainty=0.35,
        weak_regions=[
            WeakRegion(
                parameter_ranges={"E": (180e9, 220e9), "nu": (0.28, 0.32), "K_I": (4e6, 6e6)},
                metric="uncertainty",
                metric_value=0.35,
                priority=1.0,
                sample_count=2,
                suggested_samples=3,
            ),
        ],
    )

    parameter_bounds = {
        "E": (150e9, 250e9),
        "nu": (0.25, 0.35),
        "K_I": (1e6, 10e6),
        "crack_length": (0.2, 0.5),
    }

    proposals = await proposer.propose_targeted(
        context=context,
        uncertainty_analysis=uncertainty,
        parameter_bounds=parameter_bounds,
        n_samples=10,
        n_valid=10,
        n_proposals=3,
    )

    # Should return proposals with parameters
    assert len(proposals) >= 1
    for proposal in proposals:
        assert proposal.parameters is not None
        assert "E" in proposal.parameters
        assert proposal.reasoning is not None


def test_adaptive_proposer_parse_response():
    """Test parsing of LLM response into proposals."""
    from piano.agents.roles.adaptive_proposer import AdaptiveProposerAgent

    proposer = AdaptiveProposerAgent()

    # Test parsing the mock response format
    response = """
    **Proposal 1**
    Parameters: E=200e9, nu=0.30, K_I=6e6, crack_length=0.35
    Target Region: High uncertainty near crack tip
    Reasoning: Testing the singularity region
    Priority: High

    **Proposal 2**
    Parameters: E=180e9, nu=0.28, K_I=4e6, crack_length=0.25
    Target Region: Low stiffness region
    Reasoning: Exploring softer materials
    Priority: Medium
    """

    proposals = proposer._parse_multiple_proposals(response, expected_count=2)

    assert len(proposals) >= 2
    assert proposals[0].parameters["E"] == 200e9
    assert proposals[0].parameters["crack_length"] == 0.35
    assert proposals[1].parameters["nu"] == 0.28


def test_orchestrator_select_informative_samples():
    """Test orchestrator sample selection with AdaptiveProposer."""
    from piano.orchestration.adaptive import AdaptiveOrchestrator, AdaptiveConfig
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AdaptiveConfig(
            base_mesh_path=Path(tmpdir) / "mesh.mesh",
            output_dir=Path(tmpdir) / "output",
            parameter_bounds={
                "E": (150e9, 250e9),
                "nu": (0.25, 0.35),
                "K_I": (1e6, 10e6),
                "crack_length": (0.2, 0.5),
            },
            use_agentic_proposer=True,
        )

        provider = MockLLMProvider(scenario="underfitting")
        orchestrator = AdaptiveOrchestrator(config, llm_provider=provider)

        # Verify proposer is initialized
        assert orchestrator.proposer is not None

        # Test that calling _select_informative_samples without evaluator raises
        # (this is expected since we haven't set up the full training pipeline)
        with pytest.raises(RuntimeError, match="Evaluator not initialized"):
            orchestrator._select_informative_samples(3)


def test_orchestrator_proposer_initialization():
    """Test that proposer is correctly initialized based on config."""
    from piano.orchestration.adaptive import AdaptiveOrchestrator, AdaptiveConfig
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # When use_agentic_proposer=False, proposer should be None
        config_disabled = AdaptiveConfig(
            base_mesh_path=Path(tmpdir) / "mesh.mesh",
            output_dir=Path(tmpdir) / "output",
            use_agentic_proposer=False,
        )
        orchestrator_disabled = AdaptiveOrchestrator(config_disabled)
        assert orchestrator_disabled.proposer is None

        # When use_agentic_proposer=True, proposer should be initialized
        config_enabled = AdaptiveConfig(
            base_mesh_path=Path(tmpdir) / "mesh2.mesh",
            output_dir=Path(tmpdir) / "output2",
            use_agentic_proposer=True,
        )
        provider = MockLLMProvider()
        orchestrator_enabled = AdaptiveOrchestrator(config_enabled, llm_provider=provider)
        assert orchestrator_enabled.proposer is not None


# =============================================================================
# 8. Visualization Demo: Agentic Loop Progress with V-Notch FEM
# =============================================================================

def _generate_vnotch_fem_data(
    n_samples: int,
    notch_depth: float = 0.3,
    notch_angle: float = 60.0,
    resolution: int = 20,
    seed: int = 42,
    output_field: str = "von_mises",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Generate V-notch FEM dataset (or synthetic if MFEM unavailable).

    Returns:
        params: (n_samples, 3) - [E, nu, traction]
        coords: (n_nodes, 2) - mesh coordinates
        triangles: (n_elements, 3) - element connectivity
        outputs: List of output arrays (von_mises or displacement)
    """
    from piano.geometry.notch import VNotchGeometry, VNotchMeshGenerator
    from piano.data.fem_generator import generate_vnotch_fem_sample, VNotchFEMConfig

    rng = np.random.default_rng(seed)

    config = VNotchFEMConfig(
        notch_depth=notch_depth,
        notch_angle=notch_angle,
        resolution=resolution,
    )

    # Generate mesh once (all samples use same mesh)
    geometry = VNotchGeometry(
        notch_depth=notch_depth,
        notch_angle=notch_angle,
    )
    mesh_gen = VNotchMeshGenerator(geometry, base_resolution=resolution)
    coords, triangles, _ = mesh_gen.generate()

    params_list = []
    outputs = []

    for i in range(n_samples):
        E = float(rng.uniform(150e9, 250e9))
        nu = float(rng.uniform(0.25, 0.35))
        traction = float(rng.uniform(50e6, 150e6))

        sample = generate_vnotch_fem_sample(E, nu, traction, config)

        if sample is not None:
            params_list.append([E, nu, traction])
            if output_field == "von_mises" and sample.von_mises is not None:
                # Von Mises stress (scalar field) -> shape (N, 1)
                outputs.append(sample.von_mises[:, np.newaxis])
            else:
                # Displacement field -> shape (N, 2)
                outputs.append(sample.displacement)

    params = np.array(params_list, dtype=np.float32)

    return params, coords.astype(np.float32), triangles, outputs


def run_agentic_loop_demo(
    n_samples: int = 10,
    epochs_per_round: int = 20,
    max_hpo_rounds: int = 3,
    output_file: str = "tests/test_outputs/agentic_vnotch_demo.png",
):
    """
    Demonstration of agentic SciML loop for V-notch problem.

    Shows the iterative HPO process:
    - Multiple HPO rounds with agent interventions
    - Loss progression across rounds
    - Error reduction over iterations
    - Final prediction vs FEM ground truth
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.tri as mtri

    print("=" * 70)
    print("PIANO: Agentic SciML Loop - V-Notch Stress Prediction")
    print("=" * 70)

    # Configuration
    notch_depth = 0.3
    notch_angle = 60.0
    output_field = "displacement"  # Vector field — PINO-compatible (output_dim=2)

    # Generate analytical GT data
    print(f"\n1. Generating {n_samples} V-notch analytical samples (displacement)...")
    params, coords, triangles, outputs = _generate_vnotch_fem_data(
        n_samples=n_samples,
        notch_depth=notch_depth,
        notch_angle=notch_angle,
        resolution=20,
        output_field=output_field,
    )
    print(f"   Mesh: {len(coords)} nodes, {len(triangles)} elements")
    print(f"   Notch: depth={notch_depth}, angle={notch_angle}°")
    print(f"   Output: {output_field} (vector field, PINO-enabled)")

    # Import training components
    from piano.surrogate.base import TransolverConfig
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory,
    )
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.roles.physicist import PhysicistAgent
    from piano.agents.base import AgentContext
    import asyncio

    # Initialize agents
    critic = HyperparameterCriticAgent()
    architect = ArchitectAgent()
    physicist = PhysicistAgent()
    provider = MockLLMProvider(scenario="underfitting")
    critic.set_llm_provider(provider)
    architect.set_llm_provider(provider)
    physicist.set_llm_provider(provider)

    # Track progress across HPO rounds
    round_results = []
    all_train_losses = []
    all_test_losses = []
    agent_actions = []

    # Notch tip position and fixed tip_weight (geometry info, not agent-tunable)
    tip = np.array([notch_depth, 0.5], dtype=np.float32)
    tip_weight_fixed = 5.0

    # Initial config — output_dim=2 (displacement vector) enables PINO loss
    current_config = TransolverConfig(
        d_model=48,
        n_layers=2,
        n_heads=4,
        slice_num=8,
        dropout=0.05,
        learning_rate=1e-4,
        optimizer_type="adamw",
        scheduler_type="cosine",
        epochs=epochs_per_round,
        patience=15,
        batch_size=4,
        output_dim=2,  # Displacement vector field (u_x, u_y)
        pino_weight=0.05,
        tip_weight=5.0,  # Upweight nodes near notch tip singularity
    )

    context = AgentContext()
    loop = asyncio.new_event_loop()

    print(f"\n2. Running {max_hpo_rounds} HPO rounds...")

    for round_idx in range(max_hpo_rounds):
        print(f"\n   --- Round {round_idx + 1}/{max_hpo_rounds} ---")

        # Train with current config (tip_coords enables singularity-aware MSE)
        trainer = SurrogateTrainer(TrainingConfig(
            surrogate_config=current_config,
            use_ensemble=True,
            n_ensemble=3,
            train_test_split=0.2,
            tip_coords=tip,
        ))
        result = trainer.train(params, [coords] * len(params), outputs)

        # Store results
        round_results.append({
            "round": round_idx + 1,
            "train_loss": result.train_loss,
            "test_loss": result.test_loss,
            "config": current_config.to_dict().copy(),
            "trainer": trainer,
            "history": result.history,
        })

        # Accumulate loss history with round markers
        round_train = result.history.get("train_loss", [])
        round_test = result.history.get("test_loss", [])
        all_train_losses.extend(round_train)
        all_test_losses.extend(round_test)

        print(f"   Train: {result.train_loss:.6f}, Test: {result.test_loss:.6f}")

        # Critic analysis
        history = TrainingHistory(
            train_losses=round_train,
            test_losses=round_test,
            epochs_completed=len(round_train),
            final_train_loss=result.train_loss,
            final_test_loss=result.test_loss,
        )

        critique = loop.run_until_complete(
            critic.analyze_training(context, history, current_config.to_dict())
        )

        action = {
            "round": round_idx + 1,
            "issue": critique.primary_issue.name,
            "severity": critique.severity,
            "arch_changes": {},
            "phys_changes": {},
        }

        if critique.should_retrain and round_idx < max_hpo_rounds - 1:
            # Get proposals from both agents
            arch_proposal = loop.run_until_complete(
                architect.propose_config(context, current_config, critique, len(params))
            )
            phys_proposal = loop.run_until_complete(
                physicist.propose_physics_config(
                    context, current_config.to_dict(), critique, history,
                    len(params), problem_type="notch", has_singularity=True
                )
            )

            action["arch_changes"] = arch_proposal.changes
            action["phys_changes"] = phys_proposal.changes

            # Apply changes; preserve tip_weight and output_dim (not agent-tunable)
            current_config = arch_proposal.config
            current_config.tip_weight = tip_weight_fixed
            current_config.output_dim = 2
            for k, v in phys_proposal.changes.items():
                if hasattr(current_config, k):
                    setattr(current_config, k, v)

            print(f"   Critic: {critique.primary_issue.name}")
            print(f"   Architect: {list(arch_proposal.changes.keys())}")
            print(f"   Physicist: {list(phys_proposal.changes.keys())}")

        agent_actions.append(action)

    loop.close()

    # Generate test prediction with final model
    print("\n3. Generating test prediction...")
    test_E, test_nu, test_traction = 200e9, 0.3, 100e6
    test_arr = np.array([[test_E, test_nu, test_traction]], dtype=np.float32)

    # Get ground truth displacement from analytical solution
    from piano.data.fem_generator import generate_vnotch_fem_sample, VNotchFEMConfig
    gt_config = VNotchFEMConfig(notch_depth=notch_depth, notch_angle=notch_angle, resolution=20)
    gt_sample = generate_vnotch_fem_sample(test_E, test_nu, test_traction, gt_config)

    if gt_sample is not None and gt_sample.displacement is not None:
        disp_gt = gt_sample.displacement  # (N, 2)
    else:
        disp_gt = outputs[0]  # fallback to first training sample

    # Displacement magnitude for display: (N,)
    disp_mag_gt = np.linalg.norm(disp_gt, axis=1) if disp_gt.ndim == 2 else disp_gt.squeeze()

    # Predictions from each round (displacement vector)
    round_predictions = []
    for rr in round_results:
        pred, _ = rr["trainer"].predict_with_uncertainty(test_arr, coords)
        if pred.ndim == 3:
            pred = pred[0]
        round_predictions.append(pred)

    final_pred = round_predictions[-1]  # (N, 2)
    disp_mag_pred = np.linalg.norm(final_pred, axis=1) if final_pred.ndim == 2 else final_pred.squeeze()
    error = np.abs(disp_mag_pred - disp_mag_gt)

    # =========================================================================
    # VISUALIZATION: 2x3 Grid showing agentic loop progress
    # =========================================================================
    print("\n4. Creating visualization...")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Get notch tip position for marking
    tip_x, tip_y = notch_depth, 0.5

    # --- Panel 1: Loss Evolution Across All Rounds ---
    ax1 = fig.add_subplot(gs[0, 0])

    epochs_total = len(all_test_losses)
    epochs_arr = np.arange(1, epochs_total + 1)

    ax1.semilogy(epochs_arr, all_train_losses, 'b-', lw=1.5, alpha=0.7, label='Train')
    ax1.semilogy(epochs_arr, all_test_losses, 'r-', lw=2, label='Test')

    # Mark round boundaries
    epoch_offset = 0
    colors = ['green', 'orange', 'purple', 'brown']
    for i, rr in enumerate(round_results):
        n_epochs = len(rr["history"].get("train_loss", []))
        if i > 0:
            ax1.axvline(epoch_offset, color=colors[i % len(colors)], ls='--', lw=1.5, alpha=0.7)
            ax1.text(epoch_offset + 1, ax1.get_ylim()[1] * 0.8, f'R{i+1}',
                     fontsize=9, color=colors[i % len(colors)], fontweight='bold')
        epoch_offset += n_epochs

    ax1.set_xlabel('Epoch (cumulative)')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Loss Evolution Across HPO Rounds', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Test Error vs HPO Round ---
    ax2 = fig.add_subplot(gs[0, 1])

    rounds = [rr["round"] for rr in round_results]
    test_losses = [rr["test_loss"] for rr in round_results]

    ax2.plot(rounds, test_losses, 'ro-', lw=2, markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(rounds, test_losses, alpha=0.3, color='red')

    for i, (r, loss) in enumerate(zip(rounds, test_losses)):
        ax2.annotate(f'{loss:.4f}', (r, loss), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    improvement = (test_losses[0] - test_losses[-1]) / test_losses[0] * 100
    ax2.set_xlabel('HPO Round')
    ax2.set_ylabel('Test Loss')
    ax2.set_title(f'Convergence: {improvement:.1f}% Improvement', fontweight='bold')
    ax2.set_xticks(rounds)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Agent Activity Timeline ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    timeline_text = "AGENT ACTIVITY LOG\n" + "=" * 40 + "\n\n"
    for action in agent_actions:
        timeline_text += f"Round {action['round']}:\n"
        timeline_text += f"  Critic: {action['issue']} ({action['severity']})\n"
        if action['arch_changes']:
            changes = ', '.join(f"{k}" for k in list(action['arch_changes'].keys())[:3])
            timeline_text += f"  Architect: {changes}\n"
        if action['phys_changes']:
            changes = ', '.join(f"{k}" for k in list(action['phys_changes'].keys())[:2])
            timeline_text += f"  Physicist: {changes}\n"
        timeline_text += "\n"

    ax3.text(0.05, 0.95, timeline_text, transform=ax3.transAxes,
             fontsize=9, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # --- Panel 4: Final Prediction (Displacement Magnitude) ---
    ax4 = fig.add_subplot(gs[1, 0])
    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)

    # Use common scale for both plots
    vmin = min(disp_mag_pred.min(), disp_mag_gt.min())
    vmax = max(disp_mag_pred.max(), disp_mag_gt.max())
    levels = np.linspace(vmin, vmax, 20)

    cf4 = ax4.tricontourf(triang, disp_mag_pred, levels=levels, cmap='viridis', extend='both')
    ax4.triplot(triang, 'k-', lw=0.1, alpha=0.15)

    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_aspect('equal')
    ax4.set_title(r'Surrogate: Displacement $|u|$', fontweight='bold')
    fig.colorbar(cf4, ax=ax4, shrink=0.7, label=r'$|u|$ [m]', format='%.1e')

    # --- Panel 5: Ground Truth (Analytical Displacement Magnitude) ---
    ax5 = fig.add_subplot(gs[1, 1])

    cf5 = ax5.tricontourf(triang, disp_mag_gt, levels=levels, cmap='viridis', extend='both')
    ax5.triplot(triang, 'k-', lw=0.1, alpha=0.15)

    ax5.set_xlim(-0.05, 1.05)
    ax5.set_ylim(-0.05, 1.05)
    ax5.set_aspect('equal')
    ax5.set_title(r'Ground Truth: Displacement $|u|$', fontweight='bold')
    fig.colorbar(cf5, ax=ax5, shrink=0.7, label=r'$|u|$ [m]', format='%.1e')

    # --- Panel 6: Displacement Error Distribution ---
    ax6 = fig.add_subplot(gs[1, 2])

    error_levels = np.linspace(0, error.max(), 20)
    cf6 = ax6.tricontourf(triang, error, levels=error_levels, cmap='Reds', extend='both')
    ax6.triplot(triang, 'k-', lw=0.1, alpha=0.15)

    ax6.set_xlim(-0.05, 1.05)
    ax6.set_ylim(-0.05, 1.05)
    ax6.set_aspect('equal')
    ax6.set_title(f'Displacement Error (mean={error.mean():.2e} m)', fontweight='bold')
    fig.colorbar(cf6, ax=ax6, shrink=0.7, label=r'$|\Delta u|$ [m]', format='%.1e')

    # Main title
    fig.suptitle(
        f'PIANO: Displacement Prediction ({max_hpo_rounds} rounds, {improvement:.1f}% improvement)',
        fontsize=14, fontweight='bold'
    )

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved visualization to: {output_path}")
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  V-Notch: depth={notch_depth}, angle={notch_angle}°")
    print(f"  Output: Displacement magnitude (vector field, PINO-enabled)")
    print(f"  Training samples: {n_samples}")
    print(f"  HPO rounds: {max_hpo_rounds}")
    print(f"  Initial test loss: {round_results[0]['test_loss']:.6f}")
    print(f"  Final test loss: {round_results[-1]['test_loss']:.6f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Mean displacement error: {error.mean():.2e} m")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PIANO: Agentic SciML V-Notch Demo")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of FEM samples (default: 10)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epochs per HPO round (default: 20)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of HPO rounds (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path")
    args = parser.parse_args()

    output = args.output or str(PROJECT_ROOT / "tests" / "test_outputs" / "agentic_vnotch_demo.png")

    run_agentic_loop_demo(
        n_samples=args.n_samples,
        epochs_per_round=args.epochs,
        max_hpo_rounds=args.rounds,
        output_file=output,
    )
