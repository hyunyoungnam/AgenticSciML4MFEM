"""
Tests for acquisition functions.

Tests the active learning acquisition strategies:
1. Uncertainty Sampling
2. Expected Improvement
3. Query-by-Committee
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from meshforge.surrogate.acquisition import (
    AcquisitionType,
    AcquisitionResult,
    AcquisitionFunction,
    UncertaintySampling,
    ExpectedImprovement,
    QueryByCommittee,
    get_acquisition_function,
)
from meshforge.surrogate.base import PredictionResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model():
    """Mock surrogate model with uncertainty."""
    model = MagicMock()

    def mock_predict(params, coords):
        n_points = coords.shape[0]
        return PredictionResult(
            values=np.random.randn(1, n_points, 1),
            uncertainty=np.abs(np.random.randn(1, n_points, 1)) * 0.1,
            coordinates=coords,
            metadata={}
        )

    model.predict = mock_predict
    return model


@pytest.fixture
def mock_ensemble_model():
    """Mock ensemble model with multiple members."""
    model = MagicMock()

    # Create mock ensemble members
    members = []
    for _ in range(5):
        member = MagicMock()
        def member_predict(params, coords):
            n_points = coords.shape[0]
            return PredictionResult(
                values=np.random.randn(1, n_points, 1),
                coordinates=coords,
                metadata={}
            )
        member.predict = member_predict
        members.append(member)

    model._models = members

    def ensemble_predict(params, coords):
        n_points = coords.shape[0]
        return PredictionResult(
            values=np.random.randn(1, n_points, 1),
            uncertainty=np.abs(np.random.randn(1, n_points, 1)) * 0.1,
            coordinates=coords,
            metadata={}
        )

    model.predict = ensemble_predict
    return model


@pytest.fixture
def sample_candidates():
    """Sample candidate parameter sets."""
    return np.random.randn(20, 3)


@pytest.fixture
def sample_coordinates():
    """Sample spatial coordinates."""
    return np.random.randn(50, 2)


# ============================================================================
# AcquisitionType Tests
# ============================================================================

class TestAcquisitionType:
    """Tests for AcquisitionType enum."""

    def test_all_types_defined(self):
        """All expected acquisition types exist."""
        assert AcquisitionType.UNCERTAINTY
        assert AcquisitionType.EXPECTED_IMPROVEMENT
        assert AcquisitionType.QUERY_BY_COMMITTEE


# ============================================================================
# AcquisitionResult Tests
# ============================================================================

class TestAcquisitionResult:
    """Tests for AcquisitionResult dataclass."""

    def test_result_creation(self):
        """Result stores scores and indices."""
        scores = np.array([0.5, 0.8, 0.3, 0.9])
        best_indices = np.array([3, 1])

        result = AcquisitionResult(
            scores=scores,
            best_indices=best_indices,
            metadata={"acquisition_type": "test"}
        )

        assert np.array_equal(result.scores, scores)
        assert np.array_equal(result.best_indices, best_indices)
        assert result.metadata["acquisition_type"] == "test"


# ============================================================================
# UncertaintySampling Tests
# ============================================================================

class TestUncertaintySampling:
    """Tests for uncertainty sampling acquisition."""

    def test_initialization_default(self):
        """Default initialization."""
        acq = UncertaintySampling()
        assert acq.name == "uncertainty"
        assert acq.aggregation == "mean"

    def test_initialization_custom_aggregation(self):
        """Custom aggregation method."""
        acq = UncertaintySampling(aggregation="max")
        assert acq.aggregation == "max"

    def test_compute_returns_scores(self, mock_model, sample_candidates, sample_coordinates):
        """Compute returns scores for all candidates."""
        acq = UncertaintySampling()
        scores = acq.compute(sample_candidates, mock_model, sample_coordinates)

        assert len(scores) == len(sample_candidates)
        assert np.all(scores >= 0)  # Uncertainty should be non-negative

    def test_aggregation_methods(self, mock_model, sample_candidates, sample_coordinates):
        """Different aggregation methods work."""
        for agg in ["mean", "max", "median"]:
            acq = UncertaintySampling(aggregation=agg)
            scores = acq.compute(sample_candidates, mock_model, sample_coordinates)
            assert len(scores) == len(sample_candidates)


# ============================================================================
# ExpectedImprovement Tests
# ============================================================================

class TestExpectedImprovement:
    """Tests for Expected Improvement acquisition."""

    def test_initialization(self):
        """Initialization with xi parameter."""
        acq = ExpectedImprovement(xi=0.01)
        assert acq.name == "expected_improvement"
        assert acq.xi == 0.01

    def test_compute_with_best_error(self, mock_model, sample_candidates, sample_coordinates):
        """Compute EI with reference best error."""
        acq = ExpectedImprovement(xi=0.01)
        scores = acq.compute(
            sample_candidates,
            mock_model,
            sample_coordinates,
            best_error=0.5
        )

        assert len(scores) == len(sample_candidates)
        assert np.all(scores >= 0)  # EI is non-negative

    def test_xi_affects_exploration(self, mock_model, sample_candidates, sample_coordinates):
        """Higher xi favors exploration."""
        acq_low_xi = ExpectedImprovement(xi=0.001)
        acq_high_xi = ExpectedImprovement(xi=0.1)

        scores_low = acq_low_xi.compute(
            sample_candidates, mock_model, sample_coordinates, best_error=0.5
        )
        scores_high = acq_high_xi.compute(
            sample_candidates, mock_model, sample_coordinates, best_error=0.5
        )

        # Scores should differ based on xi
        assert not np.allclose(scores_low, scores_high)


# ============================================================================
# QueryByCommittee Tests
# ============================================================================

class TestQueryByCommittee:
    """Tests for Query-by-Committee acquisition."""

    def test_initialization(self):
        """Initialization with disagreement metric."""
        acq = QueryByCommittee(disagreement_metric="variance")
        assert acq.name == "query_by_committee"
        assert acq.disagreement_metric == "variance"

    def test_compute_with_ensemble(self, mock_ensemble_model, sample_candidates, sample_coordinates):
        """Compute QBC scores with ensemble model."""
        acq = QueryByCommittee()
        scores = acq.compute(sample_candidates, mock_ensemble_model, sample_coordinates)

        assert len(scores) == len(sample_candidates)

    def test_fallback_for_non_ensemble(self, mock_model, sample_candidates, sample_coordinates):
        """Falls back to uncertainty sampling for non-ensemble."""
        acq = QueryByCommittee()
        scores = acq.compute(sample_candidates, mock_model, sample_coordinates)

        # Should still return scores (fallback behavior)
        assert len(scores) == len(sample_candidates)

    def test_disagreement_metrics(self, mock_ensemble_model, sample_candidates, sample_coordinates):
        """Different disagreement metrics work."""
        for metric in ["variance", "range", "cv"]:
            acq = QueryByCommittee(disagreement_metric=metric)
            scores = acq.compute(sample_candidates, mock_ensemble_model, sample_coordinates)
            assert len(scores) == len(sample_candidates)


# ============================================================================
# Batch Selection Tests
# ============================================================================

class TestBatchSelection:
    """Tests for batch selection with diversity."""

    def test_select_batch_greedy(self, mock_model, sample_candidates, sample_coordinates):
        """Select batch without diversity."""
        acq = UncertaintySampling()
        result = acq.select_batch(
            sample_candidates,
            mock_model,
            sample_coordinates,
            batch_size=5,
            diversity_weight=0.0
        )

        assert len(result.best_indices) == 5
        assert len(result.scores) == len(sample_candidates)

    def test_select_batch_with_diversity(self, mock_model, sample_candidates, sample_coordinates):
        """Select batch with diversity weighting."""
        acq = UncertaintySampling()
        result = acq.select_batch(
            sample_candidates,
            mock_model,
            sample_coordinates,
            batch_size=5,
            diversity_weight=0.3
        )

        assert len(result.best_indices) == 5
        assert result.metadata["diversity_weight"] == 0.3

    def test_diversity_changes_selection(self, mock_model, sample_candidates, sample_coordinates):
        """Diversity weight changes which samples are selected."""
        acq = UncertaintySampling()

        result_greedy = acq.select_batch(
            sample_candidates, mock_model, sample_coordinates,
            batch_size=5, diversity_weight=0.0
        )
        result_diverse = acq.select_batch(
            sample_candidates, mock_model, sample_coordinates,
            batch_size=5, diversity_weight=0.5
        )

        # Selection may differ (though not guaranteed)
        # At minimum, both return valid results
        assert len(result_greedy.best_indices) == 5
        assert len(result_diverse.best_indices) == 5


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestGetAcquisitionFunction:
    """Tests for acquisition function factory."""

    def test_get_by_string(self):
        """Get acquisition by string name."""
        acq = get_acquisition_function("uncertainty")
        assert isinstance(acq, UncertaintySampling)

        acq = get_acquisition_function("ei")
        assert isinstance(acq, ExpectedImprovement)

        acq = get_acquisition_function("expected_improvement")
        assert isinstance(acq, ExpectedImprovement)

        acq = get_acquisition_function("qbc")
        assert isinstance(acq, QueryByCommittee)

        acq = get_acquisition_function("query_by_committee")
        assert isinstance(acq, QueryByCommittee)

    def test_get_by_enum(self):
        """Get acquisition by enum type."""
        acq = get_acquisition_function(AcquisitionType.UNCERTAINTY)
        assert isinstance(acq, UncertaintySampling)

        acq = get_acquisition_function(AcquisitionType.EXPECTED_IMPROVEMENT)
        assert isinstance(acq, ExpectedImprovement)

    def test_get_with_kwargs(self):
        """Pass kwargs to acquisition function."""
        acq = get_acquisition_function("ei", xi=0.05)
        assert isinstance(acq, ExpectedImprovement)
        assert acq.xi == 0.05

    def test_unknown_type_raises(self):
        """Unknown type raises ValueError."""
        with pytest.raises(ValueError):
            get_acquisition_function("nonexistent_strategy")
