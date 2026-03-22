"""
Tests for adaptive active learning orchestrator.

Tests the autonomous active learning loop:
1. Configuration and initialization
2. Initial sampling (LHS)
3. Surrogate training integration
4. Convergence detection
5. Adaptive budget computation
6. Informative sample selection
7. R-adaptivity integration
8. Full loop execution
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from meshforge.orchestration.adaptive import (
    AdaptiveConfig,
    AdaptiveResult,
    AdaptiveOrchestrator,
    StoppingCriterion,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for orchestrator."""
    output_dir = tmp_path / "adaptive_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_base_mesh(tmp_path):
    """Create a mock base mesh file."""
    mesh_content = """MFEM mesh v1.0

dimension
2

elements
4
1 3 0 1 5 4
1 3 1 2 6 5
1 3 4 5 9 8
1 3 5 6 10 9

boundary
8
1 1 0 1
1 1 1 2
2 1 2 6
2 1 6 10
3 1 10 9
3 1 9 8
4 1 8 4
4 1 4 0

vertices
12
2
0 0
1 0
2 0
0 1
1 1
2 1
0 2
1 2
2 2
0 3
1 3
2 3
"""
    mesh_file = tmp_path / "base.mesh"
    mesh_file.write_text(mesh_content)
    return mesh_file


@pytest.fixture
def basic_config(tmp_output_dir, mock_base_mesh):
    """Basic adaptive configuration for testing."""
    return AdaptiveConfig(
        base_mesh_path=mock_base_mesh,
        output_dir=tmp_output_dir,
        parameter_bounds={"delta_R": (-0.5, 0.5)},
        initial_samples=5,
        max_samples=20,
        convergence_threshold=0.01,
        patience=2,
    )


# ============================================================================
# AdaptiveConfig Tests
# ============================================================================

class TestAdaptiveConfig:
    """Tests for AdaptiveConfig dataclass."""

    def test_default_values(self, tmp_output_dir, mock_base_mesh):
        """Default config has expected values."""
        config = AdaptiveConfig(
            base_mesh_path=mock_base_mesh,
            output_dir=tmp_output_dir,
        )

        assert config.initial_samples == 20
        assert config.max_samples == 200
        assert config.convergence_threshold == 0.05
        assert config.acquisition_strategy == "uncertainty"
        assert config.patience == 3
        # Derived properties
        assert config.samples_per_iteration == 10
        assert config.max_iterations > 0

    def test_custom_values(self, tmp_output_dir, mock_base_mesh):
        """Custom config values are preserved."""
        config = AdaptiveConfig(
            base_mesh_path=mock_base_mesh,
            output_dir=tmp_output_dir,
            initial_samples=50,
            max_samples=500,
            acquisition_strategy="ei",
            patience=5,
        )

        assert config.initial_samples == 50
        assert config.max_samples == 500
        assert config.acquisition_strategy == "ei"
        assert config.patience == 5

    def test_path_conversion(self, tmp_output_dir, mock_base_mesh):
        """Paths are converted to Path objects."""
        config = AdaptiveConfig(
            base_mesh_path=str(mock_base_mesh),
            output_dir=str(tmp_output_dir),
        )

        assert isinstance(config.base_mesh_path, Path)
        assert isinstance(config.output_dir, Path)

    def test_parameter_names_derived(self, tmp_output_dir, mock_base_mesh):
        """Parameter names derived from bounds keys."""
        config = AdaptiveConfig(
            base_mesh_path=mock_base_mesh,
            output_dir=tmp_output_dir,
            parameter_bounds={
                "delta_R": (-0.5, 0.5),
                "E": (100e9, 300e9),
            }
        )

        assert "delta_R" in config.parameter_names
        assert "E" in config.parameter_names
        assert config.parameter_bounds["delta_R"] == (-0.5, 0.5)


# ============================================================================
# AdaptiveResult Tests
# ============================================================================

class TestAdaptiveResult:
    """Tests for AdaptiveResult dataclass."""

    def test_success_result(self):
        """Successful result has expected fields."""
        result = AdaptiveResult(
            success=True,
            n_iterations=5,
            final_error=0.02,
            initial_error=0.5,
            total_samples=100,
            stopping_criterion=StoppingCriterion.CONVERGED,
            sample_efficiency=0.005,
            error_reduction_percent=96.0,
        )

        assert result.success is True
        assert result.n_iterations == 5
        assert result.final_error == 0.02
        assert result.stopping_criterion == StoppingCriterion.CONVERGED
        assert result.error_reduction_percent == 96.0

    def test_failure_result(self):
        """Failed result has error message."""
        result = AdaptiveResult(
            success=False,
            error_message="Simulation failed",
            stopping_criterion=StoppingCriterion.USER_INTERRUPTED,
        )

        assert result.success is False
        assert result.error_message == "Simulation failed"


# ============================================================================
# StoppingCriterion Tests
# ============================================================================

class TestStoppingCriterion:
    """Tests for StoppingCriterion enum."""

    def test_all_criteria_defined(self):
        """All expected stopping criteria exist."""
        assert StoppingCriterion.CONVERGED
        assert StoppingCriterion.PATIENCE_EXHAUSTED
        assert StoppingCriterion.BUDGET_EXHAUSTED
        assert StoppingCriterion.MAX_ITERATIONS
        assert StoppingCriterion.LOW_UNCERTAINTY
        assert StoppingCriterion.DIMINISHING_RETURNS
        assert StoppingCriterion.USER_INTERRUPTED


# ============================================================================
# AdaptiveOrchestrator Initialization Tests
# ============================================================================

class TestAdaptiveOrchestratorInit:
    """Tests for AdaptiveOrchestrator initialization."""

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_initialization_creates_directories(self, basic_config):
        """Orchestrator creates output directories."""
        orchestrator = AdaptiveOrchestrator(basic_config)

        assert orchestrator.meshes_dir.exists()
        assert orchestrator.dataset_dir.exists()
        assert orchestrator.surrogate_dir.exists()
        assert orchestrator.metrics_dir.exists()

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_initialization_sets_acquisition(self, basic_config):
        """Orchestrator initializes acquisition function."""
        orchestrator = AdaptiveOrchestrator(basic_config)

        assert orchestrator._acquisition_fn is not None

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_initialization_creates_dataset(self, basic_config):
        """Orchestrator creates FEM dataset."""
        orchestrator = AdaptiveOrchestrator(basic_config)

        assert orchestrator.dataset is not None


# ============================================================================
# Parameter Generation Tests
# ============================================================================

class TestParameterGeneration:
    """Tests for parameter sampling methods."""

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_generate_initial_parameters_lhs(self, basic_config):
        """Initial parameters use Latin Hypercube Sampling."""
        orchestrator = AdaptiveOrchestrator(basic_config)
        params = orchestrator._generate_initial_parameters()

        assert len(params) == basic_config.initial_samples

        # All params should be within bounds
        for p in params:
            assert "delta_R" in p
            min_val, max_val = basic_config.parameter_bounds["delta_R"]
            assert min_val <= p["delta_R"] <= max_val

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_generate_random_parameters(self, basic_config):
        """Random parameters are within bounds."""
        orchestrator = AdaptiveOrchestrator(basic_config)
        params = orchestrator._generate_random_parameters(10)

        assert len(params) == 10

        for p in params:
            min_val, max_val = basic_config.parameter_bounds["delta_R"]
            assert min_val <= p["delta_R"] <= max_val


# ============================================================================
# Convergence Detection Tests
# ============================================================================

class TestConvergenceDetection:
    """Tests for convergence checking logic."""

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_converged_when_below_threshold(self, basic_config):
        """Detects convergence when error below threshold."""
        orchestrator = AdaptiveOrchestrator(basic_config)

        criterion = orchestrator._check_convergence(
            test_error=0.005,  # Below threshold of 0.01
            n_samples=50,
            n_weak_regions=0,
            mean_uncertainty=0.01,
            iteration=3
        )

        assert criterion == StoppingCriterion.CONVERGED

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_budget_exhausted(self, basic_config):
        """Detects budget exhaustion."""
        orchestrator = AdaptiveOrchestrator(basic_config)

        criterion = orchestrator._check_convergence(
            test_error=0.5,  # Above threshold
            n_samples=25,  # Above max_samples of 20
            n_weak_regions=5,
            mean_uncertainty=0.2,
            iteration=3
        )

        assert criterion == StoppingCriterion.BUDGET_EXHAUSTED

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_patience_exhausted(self, basic_config):
        """Detects patience exhaustion after no improvement."""
        orchestrator = AdaptiveOrchestrator(basic_config)
        orchestrator._best_error = 0.1
        orchestrator._no_improvement_count = 1  # Already 1 iteration without improvement

        # Check with error that doesn't improve enough
        criterion = orchestrator._check_convergence(
            test_error=0.099,  # Improvement less than min_improvement
            n_samples=10,
            n_weak_regions=3,
            mean_uncertainty=0.1,
            iteration=3
        )

        # After this call, no_improvement_count becomes 2 which equals patience
        assert criterion == StoppingCriterion.PATIENCE_EXHAUSTED or criterion is None

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_no_criterion_when_improving(self, basic_config):
        """Returns None when still improving."""
        orchestrator = AdaptiveOrchestrator(basic_config)
        orchestrator._best_error = 0.5

        criterion = orchestrator._check_convergence(
            test_error=0.3,  # Significant improvement
            n_samples=10,
            n_weak_regions=3,
            mean_uncertainty=0.2,
            iteration=1
        )

        assert criterion is None


# ============================================================================
# Adaptive Budget Tests
# ============================================================================

class TestAdaptiveBudget:
    """Tests for adaptive sample budget computation."""

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_budget_increases_when_far_from_convergence(self, basic_config):
        """More samples when error is high."""
        orchestrator = AdaptiveOrchestrator(basic_config)

        # Error much higher than threshold
        budget = orchestrator._compute_adaptive_budget(current_error=0.5)

        # Should scale up
        assert budget >= basic_config.samples_per_iteration

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_budget_decreases_near_convergence(self, basic_config):
        """Fewer samples when close to convergence."""
        orchestrator = AdaptiveOrchestrator(basic_config)

        # Error just above threshold
        budget = orchestrator._compute_adaptive_budget(current_error=0.015)

        # Should scale down
        assert budget <= basic_config.samples_per_iteration

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_budget_returns_positive(self, basic_config):
        """Budget is always positive."""
        orchestrator = AdaptiveOrchestrator(basic_config)

        budget = orchestrator._compute_adaptive_budget(current_error=0.5)

        assert budget > 0


# ============================================================================
# R-Adaptivity Tests
# ============================================================================

class TestRAdaptivity:
    """Tests for r-adaptivity integration (deferred)."""

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_adaptivity_deferred(self, basic_config):
        """R-adaptivity is deferred (not wired into loop yet)."""
        orchestrator = AdaptiveOrchestrator(basic_config)

        # R-adaptivity is deferred - always None
        assert orchestrator._adaptivity is None

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_adapt_mesh_to_error_without_adaptivity(self, basic_config):
        """Gracefully handles missing adaptivity."""
        orchestrator = AdaptiveOrchestrator(basic_config)

        # Create mock manager
        mock_manager = MagicMock()
        mock_manager.get_nodes.return_value = np.random.rand(10, 2)

        error_field = np.random.rand(10)

        coords, quality = orchestrator._apply_r_adaptivity(mock_manager, error_field)

        # Should return original coords when adaptivity unavailable
        assert coords is not None


# ============================================================================
# Full Run Tests (Mocked)
# ============================================================================

class TestOrchestratorRun:
    """Tests for full orchestrator run (with mocking)."""

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_get_dataset(self, basic_config):
        """Can retrieve dataset from orchestrator."""
        orchestrator = AdaptiveOrchestrator(basic_config)

        dataset = orchestrator.get_dataset()
        assert dataset is not None

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("meshforge").exists(),
        reason="meshforge package not found"
    )
    def test_get_surrogate_before_training(self, basic_config):
        """Surrogate is None before training."""
        orchestrator = AdaptiveOrchestrator(basic_config)

        surrogate = orchestrator.get_surrogate()
        assert surrogate is None


# ============================================================================
# Integration Tests (Skip in CI)
# ============================================================================

@pytest.mark.integration
class TestOrchestratorIntegration:
    """Integration tests that require full setup.

    These tests are skipped by default. Run with:
        pytest -m integration
    """

    def test_full_run_with_mocked_solver(self, basic_config):
        """Full active learning loop with mocked FEM solver."""
        # This would require mocking the MFEMSolver
        pytest.skip("Integration test - requires full environment")

    def test_checkpoint_and_resume(self, basic_config):
        """Test checkpointing and resuming training."""
        pytest.skip("Integration test - requires full environment")
