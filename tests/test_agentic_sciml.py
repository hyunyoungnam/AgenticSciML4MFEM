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
        responses = {
            "underfitting": """
REASONING: The crack tip singularity (1/sqrt(r)) requires high model capacity. Current architecture is too shallow to capture sharp stress gradients. Increasing depth and width, plus using SiLU activation which handles sharp transitions better than GELU.

CHANGES:
- d_model: 256 (increased from 64 for more capacity)
- n_layers: 6 (deeper to capture complex patterns)
- n_heads: 8 (more attention heads)
- slice_num: 32 (finer spatial resolution)
- activation: silu (better for sharp gradients)
- learning_rate: 1e-3 (faster initial learning)
- dropout: 0.05 (light regularization)
- pino_weight: 0.05 (reduced to allow singularity learning)

EXPECTED_IMPACT: Model should better capture crack tip stress concentration. Expect significant error reduction near tip.

CONFIDENCE: high
""",
            "overfitting": """
REASONING: Need to regularize and reduce capacity to prevent memorization.

CHANGES:
- d_model: 128 (reduced)
- dropout: 0.2 (added regularization)
- pino_weight: 0.2 (physics constraints)
- learning_rate: 5e-4 (reduced)

CONFIDENCE: high
""",
            "slow_convergence": """
REASONING: Need faster learning dynamics.

CHANGES:
- learning_rate: 1e-3 (increased)
- scheduler_type: cosine
- optimizer_type: adamw

CONFIDENCE: medium
""",
            "stable": """
REASONING: Minor tuning only.

CHANGES:
- epochs: 100

CONFIDENCE: medium
""",
        }
        return responses.get(self.scenario, responses["underfitting"])

    def _physicist_response(self) -> str:
        """Generate mock physicist response for physics loss configuration."""
        responses = {
            "underfitting": """
PHYSICS_DIAGNOSIS: The physics loss is not effectively enforcing equilibrium near the crack tip. The 1/sqrt(r) singularity causes high residuals that dominate the loss, but current weights are too low to guide learning. Need to increase physics enforcement gradually.

CHANGES:
- pino_weight: 0.15 (increased to enforce energy consistency)
- pino_eq_weight: 0.2 (higher to enforce equilibrium near singularity)

REASONING: For crack tip singularities, equilibrium enforcement is critical. The stress field must satisfy div(sigma)=0 everywhere including the singular region. Higher eq_weight helps the model learn the correct asymptotic behavior.

EXPECTED_IMPACT: Better capture of crack tip stress field. Equilibrium residual should decrease near tip.

CONFIDENCE: high
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
# 7. Visualization Demo with Crack Contour Plots
# =============================================================================

def run_agentic_loop_demo(
    n_samples: int = 8,
    epochs: int = 30,
    output_file: str = "tests/test_outputs/agentic_crack_demo.png",
):
    """
    Demonstration of agentic SciML loop for crack problems.

    Generates visualization with:
    - Displacement contour on crack mesh
    - Error concentration at crack tip
    - Training convergence before/after HPO
    - Hyperparameter changes
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.tri as mtri

    print("=" * 70)
    print("Agentic SciML Loop: Static Crack Problem")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Generate crack mesh
    print("\n1. Generating crack mesh...")
    crack_length = 0.35
    coords, triangles = _generate_crack_mesh(
        n_points=500, crack_length=crack_length, rng=rng
    )
    tip_x, tip_y = crack_length, 0.5
    print(f"   Mesh: {len(coords)} nodes, {len(triangles)} triangles")
    print(f"   Crack length: {crack_length}, tip at ({tip_x:.2f}, {tip_y:.2f})")

    # Generate training samples
    print(f"\n2. Generating {n_samples} training samples (Williams expansion)...")
    params_list = []
    outputs = []
    for i in range(n_samples):
        p = {
            "E": float(rng.uniform(150e9, 250e9)),
            "nu": float(rng.uniform(0.25, 0.35)),
            "K_I": float(rng.uniform(2e6, 8e6)),
            "crack_length": crack_length,
        }
        params_list.append(p)
        disp = _williams_displacement(coords, p, tip_x=tip_x, tip_y=tip_y)
        outputs.append(disp)

    params = np.array([[p["E"], p["nu"], p["K_I"], p["crack_length"]]
                       for p in params_list], dtype=np.float32)

    # Import training components
    from piano.surrogate.base import TransolverConfig
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory,
    )
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.base import AgentContext
    import asyncio

    # Initial config (intentionally weak for crack singularity)
    print("\n3. Training with INITIAL config (weak for singularity)...")
    initial_config = TransolverConfig(
        d_model=64,       # Too small
        n_layers=2,       # Too shallow
        n_heads=4,
        slice_num=8,      # Too coarse
        dropout=0.0,
        learning_rate=1e-4,  # Too slow
        optimizer_type="adam",
        scheduler_type="plateau",
        epochs=epochs,
        patience=20,
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
    print(f"   Initial: train={result1.train_loss:.6f}, test={result1.test_loss:.6f}")

    # Critic analysis
    print("\n4. Critic analyzing training...")
    history = TrainingHistory(
        train_losses=result1.history.get("train_loss", []),
        test_losses=result1.history.get("test_loss", []),
        epochs_completed=len(result1.history.get("train_loss", [])),
        final_train_loss=result1.train_loss,
        final_test_loss=result1.test_loss,
    )

    from piano.agents.roles.physicist import PhysicistAgent

    critic = HyperparameterCriticAgent()
    architect = ArchitectAgent()
    physicist = PhysicistAgent()
    provider = MockLLMProvider(scenario="underfitting")
    critic.set_llm_provider(provider)
    architect.set_llm_provider(provider)
    physicist.set_llm_provider(provider)

    context = AgentContext()
    loop = asyncio.new_event_loop()

    critique = loop.run_until_complete(
        critic.analyze_training(context, history, initial_config.to_dict())
    )
    print(f"   Diagnosis: {critique.primary_issue.name} ({critique.severity})")

    # Architect proposal (architecture + optimizer)
    print("\n5. Architect proposing architecture changes...")
    arch_proposal = loop.run_until_complete(
        architect.propose_config(context, initial_config, critique, n_samples)
    )
    print(f"   Architecture: {arch_proposal.changes}")

    # Physicist proposal (physics loss config)
    print("\n6. Physicist proposing physics loss changes...")
    phys_proposal = loop.run_until_complete(
        physicist.propose_physics_config(
            context, initial_config.to_dict(), critique, history,
            n_samples, problem_type="crack", has_singularity=True
        )
    )
    print(f"   Physics: {phys_proposal.changes}")

    # Merge proposals
    proposal = arch_proposal  # Use architect's config as base
    for k, v in phys_proposal.changes.items():
        if k.startswith("pino"):
            proposal.changes[k] = v

    loop.close()

    # Retrain with optimized config
    print("\n7. Retraining with OPTIMIZED config...")
    new_config = proposal.config

    trainer2 = SurrogateTrainer(TrainingConfig(
        surrogate_config=new_config,
        use_ensemble=True,
        n_ensemble=3,
        train_test_split=0.2,
    ))
    result2 = trainer2.train(params, [coords] * n_samples, outputs)
    print(f"   After HPO: train={result2.train_loss:.6f}, test={result2.test_loss:.6f}")

    # Store physics changes for visualization
    physics_changes = phys_proposal.changes

    # Test prediction
    print("\n9. Generating test prediction...")
    test_params = {"E": 200e9, "nu": 0.3, "K_I": 5e6, "crack_length": crack_length}
    test_arr = np.array([[test_params["E"], test_params["nu"],
                          test_params["K_I"], test_params["crack_length"]]],
                        dtype=np.float32)

    disp_gt = _williams_displacement(coords, test_params, tip_x=tip_x, tip_y=tip_y)
    disp_mag_gt = np.linalg.norm(disp_gt, axis=1)

    pred, unc = trainer2.predict_with_uncertainty(test_arr, coords)
    if pred.ndim == 3:
        pred = pred[0]
    disp_mag_pred = np.linalg.norm(pred, axis=-1)

    error = np.abs(disp_mag_pred - disp_mag_gt)

    # Visualization
    print("\n10. Creating 6-panel visualization...")

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)

    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)

    def _contour(ax, values, title, cmap, vmin=None, vmax=None):
        v0 = values.min() if vmin is None else vmin
        v1 = values.max() if vmax is None else vmax
        levels = np.linspace(v0, v1, 20)
        cf = ax.tricontourf(triang, values, levels=levels, cmap=cmap, extend='both')
        ax.triplot(triang, 'k-', lw=0.1, alpha=0.2)
        # Mark crack
        ax.plot([0, tip_x], [0.5, tip_y], 'r-', lw=2, label='Crack')
        ax.plot(tip_x, tip_y, 'r*', markersize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        return cf

    # Row 1: Contour plots
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = _contour(ax1, disp_mag_pred, "Predicted |u| (Displacement)", "jet")
    fig.colorbar(cf1, ax=ax1, shrink=0.7, label="|u| [m]")

    ax2 = fig.add_subplot(gs[0, 1])
    cf2 = _contour(ax2, disp_mag_gt, "Ground Truth |u| (Williams)", "jet",
                   vmin=disp_mag_pred.min(), vmax=disp_mag_pred.max())
    fig.colorbar(cf2, ax=ax2, shrink=0.7, label="|u| [m]")

    ax3 = fig.add_subplot(gs[0, 2])
    cf3 = _contour(ax3, error, "Prediction Error (near tip = hard)", "Reds")
    fig.colorbar(cf3, ax=ax3, shrink=0.7, label="Error [m]")
    ax3.text(0.02, 0.98, f"Mean: {error.mean():.2e}\nMax: {error.max():.2e}",
             transform=ax3.transAxes, fontsize=8, va='top',
             bbox=dict(facecolor='white', alpha=0.8))

    # Row 2: Analysis
    ax4 = fig.add_subplot(gs[1, 0])
    if result1.history and result2.history:
        ep1 = np.arange(1, len(result1.history["test_loss"]) + 1)
        ep2 = np.arange(1, len(result2.history["test_loss"]) + 1)
        ax4.semilogy(ep1, result1.history["test_loss"], 'r--', lw=1.5,
                     alpha=0.6, label="Initial (test)")
        ax4.semilogy(ep2, result2.history["test_loss"], 'b-', lw=2,
                     label="After HPO (test)")
        ax4.semilogy(ep2, result2.history["train_loss"], 'g--', lw=1.5,
                     alpha=0.6, label="After HPO (train)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss (log)")
    ax4.set_title("Training Convergence", fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Config changes
    ax5 = fig.add_subplot(gs[1, 1])
    cfg_params = ["d_model", "n_layers", "slice_num", "learning_rate", "dropout"]
    x = np.arange(len(cfg_params))
    init_v = [initial_config.to_dict().get(p, 0) for p in cfg_params]
    new_v = [new_config.to_dict().get(p, 0) for p in cfg_params]
    max_v = [max(abs(i), abs(n), 1e-10) for i, n in zip(init_v, new_v)]
    init_n = [i / m for i, m in zip(init_v, max_v)]
    new_n = [n / m for n, m in zip(new_v, max_v)]

    w = 0.35
    ax5.bar(x - w/2, init_n, w, label="Initial", color="steelblue")
    ax5.bar(x + w/2, new_n, w, label="After HPO", color="coral")
    ax5.set_xticks(x)
    ax5.set_xticklabels(cfg_params, rotation=20, ha="right")
    ax5.set_title("Hyperparameter Changes", fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(True, axis="y", alpha=0.3)

    # Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    improvement = (result1.test_loss - result2.test_loss) / result1.test_loss * 100

    # Separate architecture and physics changes for display
    arch_changes_str = chr(10).join(
        f'  {k}: {v}' for k, v in list(arch_proposal.changes.items())[:3]
        if not k.startswith('pino')
    )
    phys_changes_str = chr(10).join(
        f'  {k}: {v}' for k, v in physics_changes.items()
    ) or "  (no changes)"

    summary = f"""
AGENTIC SCIML: 3-AGENT HPO
{'='*44}

Problem: Edge crack with 1/sqrt(r) singularity
Crack: a={crack_length:.2f}, Tip: ({tip_x:.2f}, {tip_y:.2f})

Critic: {critique.primary_issue.name} ({critique.severity})

Architect (architecture):
{arch_changes_str}

Physicist (physics loss):
{phys_changes_str}

Performance:
  Initial loss: {result1.test_loss:.6f}
  Final loss:   {result2.test_loss:.6f}
  Change:       {improvement:+.1f}%

Error: mean={error.mean():.2e}, max={error.max():.2e}
"""
    ax6.text(0.02, 0.98, summary, transform=ax6.transAxes,
             fontsize=9, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Agentic SciML: 3-Agent HPO (Critic + Architect + Physicist)",
                 fontsize=14, fontweight="bold")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved visualization to: {output_path}")
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Crack length:      {crack_length}")
    print(f"  Training samples:  {n_samples}")
    print(f"  Initial test loss: {result1.test_loss:.6f}")
    print(f"  Final test loss:   {result2.test_loss:.6f}")
    print(f"  Improvement:       {improvement:.1f}%")
    print(f"  Mean error:        {error.mean():.2e} m")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agentic SciML: Crack Problem")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of samples (default: 8)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs (default: 30)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path")
    args = parser.parse_args()

    output = args.output or str(PROJECT_ROOT / "tests" / "test_outputs" / "agentic_crack_demo.png")

    run_agentic_loop_demo(
        n_samples=args.n_samples,
        epochs=args.epochs,
        output_file=output,
    )
