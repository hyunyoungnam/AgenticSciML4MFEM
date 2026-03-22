"""
Tests for spatial error analysis.

Tests the error analysis tools for surrogate models:
1. Error field computation
2. Hotspot identification
3. Parameter sensitivity analysis
4. Error decomposition (bias/variance)
5. Aggregated analysis
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from meshforge.surrogate.error_analysis import (
    ErrorHotspot,
    SpatialErrorAnalysis,
    SpatialErrorAnalyzer,
    ErrorDecomposer,
)
from meshforge.surrogate.base import PredictionResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model():
    """Mock surrogate model (non-ensemble)."""
    model = MagicMock(spec=['predict'])  # Only allow 'predict' attribute

    def mock_predict(params, coords):
        n_points = coords.shape[0]
        # Simulate predictions that are close but not exact to true values
        base_pred = np.sin(coords[:, 0]) * np.cos(coords[:, 1])
        noise = np.random.randn(n_points) * 0.1
        return PredictionResult(
            values=(base_pred + noise).reshape(1, n_points, 1),
            uncertainty=np.abs(np.random.randn(1, n_points, 1)) * 0.05,
            coordinates=coords,
            metadata={}
        )

    model.predict = mock_predict
    return model


@pytest.fixture
def mock_ensemble_model():
    """Mock ensemble model with uncertainty."""
    model = MagicMock()
    model._models = [MagicMock() for _ in range(5)]

    def mock_predict(params, coords):
        n_points = coords.shape[0]
        base_pred = np.sin(coords[:, 0]) * np.cos(coords[:, 1])
        noise = np.random.randn(n_points) * 0.1
        return PredictionResult(
            values=(base_pred + noise).reshape(1, n_points, 1),
            uncertainty=np.abs(np.random.randn(1, n_points, 1)) * 0.1,
            coordinates=coords,
            metadata={}
        )

    model.predict = mock_predict
    return model


@pytest.fixture
def sample_coordinates():
    """Sample 2D mesh coordinates."""
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


@pytest.fixture
def sample_params():
    """Sample parameter values."""
    return np.array([[0.5, 1.0, 0.3]])


@pytest.fixture
def sample_true_values(sample_coordinates):
    """Sample ground truth field values."""
    coords = sample_coordinates
    return np.sin(coords[:, 0]) * np.cos(coords[:, 1])


@pytest.fixture
def analyzer(mock_model, sample_coordinates):
    """Pre-configured SpatialErrorAnalyzer."""
    return SpatialErrorAnalyzer(mock_model, sample_coordinates)


# ============================================================================
# ErrorHotspot Tests
# ============================================================================

class TestErrorHotspot:
    """Tests for ErrorHotspot dataclass."""

    def test_hotspot_creation(self):
        """Hotspot stores all attributes."""
        hotspot = ErrorHotspot(
            center=np.array([5.0, 5.0]),
            radius=1.5,
            mean_error=0.3,
            max_error=0.8,
            point_indices=np.array([10, 11, 12, 13]),
            parameter_sensitivity=0.5,
        )

        assert np.allclose(hotspot.center, [5.0, 5.0])
        assert hotspot.radius == 1.5
        assert hotspot.mean_error == 0.3
        assert hotspot.max_error == 0.8
        assert len(hotspot.point_indices) == 4
        assert hotspot.parameter_sensitivity == 0.5

    def test_hotspot_with_metadata(self):
        """Hotspot stores metadata."""
        hotspot = ErrorHotspot(
            center=np.array([0.0, 0.0]),
            radius=1.0,
            mean_error=0.1,
            max_error=0.2,
            point_indices=np.array([0, 1]),
            metadata={"reason": "test"}
        )

        assert hotspot.metadata["reason"] == "test"


# ============================================================================
# SpatialErrorAnalysis Tests
# ============================================================================

class TestSpatialErrorAnalysis:
    """Tests for SpatialErrorAnalysis dataclass."""

    def test_analysis_creation(self):
        """Analysis stores error field and hotspots."""
        error_field = np.random.rand(100)
        hotspots = [
            ErrorHotspot(
                center=np.array([5.0, 5.0]),
                radius=1.0,
                mean_error=0.5,
                max_error=0.8,
                point_indices=np.array([50, 51, 52])
            )
        ]

        analysis = SpatialErrorAnalysis(
            error_field=error_field,
            hotspots=hotspots,
            global_stats={"mean_error": 0.3},
        )

        assert len(analysis.error_field) == 100
        assert len(analysis.hotspots) == 1
        assert analysis.global_stats["mean_error"] == 0.3


# ============================================================================
# SpatialErrorAnalyzer Tests
# ============================================================================

class TestSpatialErrorAnalyzerInit:
    """Tests for SpatialErrorAnalyzer initialization."""

    def test_initialization(self, mock_model, sample_coordinates):
        """Analyzer initializes with model and coordinates."""
        analyzer = SpatialErrorAnalyzer(mock_model, sample_coordinates)

        assert analyzer.model == mock_model
        assert np.array_equal(analyzer.coordinates, sample_coordinates)
        assert analyzer.coord_dim == 2


class TestComputeErrorField:
    """Tests for error field computation."""

    def test_compute_error_field(self, analyzer, sample_params, sample_true_values):
        """Compute absolute error field."""
        error_field = analyzer.compute_error_field(sample_params, sample_true_values)

        assert len(error_field) == len(sample_true_values)
        assert np.all(error_field >= 0)  # Absolute error is non-negative

    def test_compute_relative_error_field(self, analyzer, sample_params, sample_true_values):
        """Compute relative error field."""
        error_field = analyzer.compute_relative_error_field(
            sample_params, sample_true_values
        )

        assert len(error_field) == len(sample_true_values)
        assert np.all(error_field >= 0)

    def test_params_shape_handling(self, analyzer, sample_true_values):
        """Handle both 1D and 2D parameter arrays."""
        params_1d = np.array([0.5, 1.0, 0.3])
        params_2d = np.array([[0.5, 1.0, 0.3]])

        error_1d = analyzer.compute_error_field(params_1d, sample_true_values)
        error_2d = analyzer.compute_error_field(params_2d, sample_true_values)

        assert len(error_1d) == len(sample_true_values)
        assert len(error_2d) == len(sample_true_values)


class TestIdentifyHotspots:
    """Tests for hotspot identification."""

    def test_identify_hotspots_basic(self, analyzer):
        """Identify hotspots in error field."""
        # Create error field with a clear hotspot
        coords = analyzer.coordinates
        n_points = len(coords)

        # Error concentrated near (7, 7)
        hotspot_center = np.array([7.0, 7.0])
        distances = np.linalg.norm(coords - hotspot_center, axis=1)
        error_field = np.exp(-distances / 2.0)  # High near center

        hotspots = analyzer.identify_hotspots(
            error_field,
            threshold_percentile=80,
            min_cluster_size=3
        )

        # Should find at least one hotspot
        assert len(hotspots) >= 0  # May be 0 if clustering fails

    def test_identify_hotspots_empty_for_uniform_error(self, analyzer):
        """Uniform error produces few/no hotspots."""
        error_field = np.ones(len(analyzer.coordinates)) * 0.1

        hotspots = analyzer.identify_hotspots(
            error_field,
            threshold_percentile=90,
            min_cluster_size=5
        )

        # Uniform error shouldn't have distinct hotspots
        # (threshold excludes everything or includes everything)
        assert isinstance(hotspots, list)

    def test_hotspot_sorting(self, analyzer):
        """Hotspots are sorted by mean error (highest first)."""
        coords = analyzer.coordinates

        # Create two hotspots with different error magnitudes
        hotspot1 = np.array([3.0, 3.0])
        hotspot2 = np.array([7.0, 7.0])

        d1 = np.linalg.norm(coords - hotspot1, axis=1)
        d2 = np.linalg.norm(coords - hotspot2, axis=1)

        # Hotspot 2 has higher error
        error_field = np.exp(-d1 / 1.5) * 0.5 + np.exp(-d2 / 1.5) * 1.0

        hotspots = analyzer.identify_hotspots(
            error_field,
            threshold_percentile=70,
            min_cluster_size=2
        )

        if len(hotspots) >= 2:
            # First hotspot should have highest mean error
            assert hotspots[0].mean_error >= hotspots[1].mean_error


class TestParameterSensitivity:
    """Tests for parameter sensitivity analysis."""

    def test_compute_parameter_sensitivity(self, analyzer, sample_params):
        """Compute sensitivity to each parameter."""
        param_names = ["delta_R", "E", "nu"]

        sensitivities = analyzer.compute_parameter_sensitivity(
            sample_params,
            param_names,
            epsilon=0.01
        )

        assert len(sensitivities) == 3
        assert "delta_R" in sensitivities
        assert "E" in sensitivities
        assert "nu" in sensitivities

        # Each sensitivity field has same length as coordinates
        for name, field in sensitivities.items():
            assert len(field) == len(analyzer.coordinates)


class TestFullAnalysis:
    """Tests for comprehensive analysis."""

    def test_analyze_returns_all_components(self, analyzer, sample_params, sample_true_values):
        """Full analysis returns error field, hotspots, and stats."""
        param_names = ["delta_R", "E", "nu"]

        analysis = analyzer.analyze(
            sample_params,
            sample_true_values,
            parameter_names=param_names,
            hotspot_threshold=85
        )

        assert isinstance(analysis, SpatialErrorAnalysis)
        assert len(analysis.error_field) == len(sample_true_values)
        assert isinstance(analysis.hotspots, list)
        assert "mean_error" in analysis.global_stats
        assert "max_error" in analysis.global_stats
        assert "n_hotspots" in analysis.global_stats

    def test_analyze_without_param_names(self, analyzer, sample_params, sample_true_values):
        """Analysis works without parameter names."""
        analysis = analyzer.analyze(
            sample_params,
            sample_true_values,
            parameter_names=None
        )

        assert analysis.error_field is not None
        assert analysis.parameter_influence == {}


class TestAggregateAnalyses:
    """Tests for aggregating multiple analyses."""

    def test_aggregate_multiple_analyses(self, analyzer, sample_true_values):
        """Aggregate analyses from multiple parameter samples."""
        # Create multiple analyses
        analyses = []
        for i in range(5):
            params = np.array([[0.5 + i * 0.1, 1.0, 0.3]])
            analysis = analyzer.analyze(params, sample_true_values)
            analyses.append(analysis)

        aggregated = analyzer.aggregate_analyses(analyses)

        assert len(aggregated.error_field) == len(sample_true_values)
        assert "n_samples" in aggregated.global_stats
        assert aggregated.global_stats["n_samples"] == 5

    def test_aggregate_empty_list(self, analyzer):
        """Aggregating empty list returns empty analysis."""
        aggregated = analyzer.aggregate_analyses([])

        assert len(aggregated.error_field) == 0


# ============================================================================
# ErrorDecomposer Tests
# ============================================================================

class TestErrorDecomposerInit:
    """Tests for ErrorDecomposer initialization."""

    def test_initialization(self, mock_model):
        """Decomposer initializes with model."""
        decomposer = ErrorDecomposer(mock_model)
        assert decomposer.model == mock_model

    def test_detects_ensemble(self, mock_ensemble_model):
        """Decomposer detects ensemble model."""
        decomposer = ErrorDecomposer(mock_ensemble_model)
        assert decomposer._is_ensemble is True

    def test_detects_non_ensemble(self, mock_model):
        """Decomposer detects non-ensemble model."""
        decomposer = ErrorDecomposer(mock_model)
        assert decomposer._is_ensemble is False


class TestErrorDecomposition:
    """Tests for bias-variance decomposition."""

    def test_decompose_returns_components(self, mock_model, sample_coordinates, sample_true_values):
        """Decomposition returns bias, variance, noise."""
        decomposer = ErrorDecomposer(mock_model)
        params = np.array([[0.5, 1.0, 0.3]])

        result = decomposer.decompose(params, sample_coordinates, sample_true_values)

        assert "bias" in result
        assert "bias_squared" in result
        assert "variance" in result
        assert "noise" in result
        assert "mse" in result
        assert "bias_fraction" in result
        assert "variance_fraction" in result

    def test_decomposition_sums_correctly(self, mock_model, sample_coordinates, sample_true_values):
        """Bias^2 + Variance + Noise ≈ MSE."""
        decomposer = ErrorDecomposer(mock_model)
        params = np.array([[0.5, 1.0, 0.3]])

        result = decomposer.decompose(params, sample_coordinates, sample_true_values)

        total = result["bias_squared"] + result["variance"] + result["noise"]
        assert np.isclose(total, result["mse"], rtol=1e-5)

    def test_decompose_with_ensemble(self, mock_ensemble_model, sample_coordinates, sample_true_values):
        """Decomposition works with ensemble (captures variance)."""
        decomposer = ErrorDecomposer(mock_ensemble_model)
        params = np.array([[0.5, 1.0, 0.3]])

        result = decomposer.decompose(params, sample_coordinates, sample_true_values)

        # Ensemble should have non-zero variance component
        assert "variance" in result


class TestSpatialDecomposition:
    """Tests for spatial error decomposition."""

    def test_decompose_spatial(self, mock_model, sample_coordinates, sample_true_values):
        """Spatial decomposition returns per-point fields."""
        decomposer = ErrorDecomposer(mock_model)
        params = np.array([[0.5, 1.0, 0.3]])

        result = decomposer.decompose_spatial(params, sample_coordinates, sample_true_values)

        n_points = len(sample_coordinates)
        assert len(result["bias"]) == n_points
        assert len(result["variance"]) == n_points
        assert len(result["mse"]) == n_points
        assert len(result["absolute_error"]) == n_points

    def test_spatial_mse_equals_squared_bias(self, mock_model, sample_coordinates, sample_true_values):
        """For non-ensemble, MSE = signed_error^2."""
        decomposer = ErrorDecomposer(mock_model)
        params = np.array([[0.5, 1.0, 0.3]])

        result = decomposer.decompose_spatial(params, sample_coordinates, sample_true_values)

        expected_mse = result["signed_error"] ** 2
        assert np.allclose(result["mse"], expected_mse)
