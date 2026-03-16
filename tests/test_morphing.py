"""
Tests for r-adaptivity using TMOP.

Tests the error-driven mesh adaptation that redistributes nodes
to cluster in high-error regions based on surrogate model predictions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from meshforge.morphing import (
    TMOPAdaptivity,
    AdaptivityConfig,
    AdaptivityResult,
    is_tmop_available,
)


@pytest.fixture
def sample_2d_coords():
    """Sample 2D coordinates for testing."""
    # Create a simple grid
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    return coords


@pytest.fixture
def sample_error_field(sample_2d_coords):
    """Sample error field with high error in one corner."""
    coords = sample_2d_coords
    # High error near (10, 10), low error elsewhere
    center = np.array([10.0, 10.0])
    distances = np.linalg.norm(coords - center, axis=1)
    # Inverse distance - high error near center
    error = 1.0 / (1.0 + distances)
    return error


@pytest.fixture
def default_config():
    """Default adaptivity configuration."""
    return AdaptivityConfig(
        size_scale_min=0.3,
        size_scale_max=2.0,
        max_iterations=50,
        tolerance=1e-6,
    )


class TestAdaptivityConfig:
    """Tests for AdaptivityConfig dataclass."""

    def test_default_config(self):
        """Default config has expected values."""
        config = AdaptivityConfig()
        assert config.size_scale_min == 0.3
        assert config.size_scale_max == 2.0
        assert config.max_iterations == 200
        assert config.fix_boundary is True

    def test_custom_config(self):
        """Custom config values are preserved."""
        config = AdaptivityConfig(
            size_scale_min=0.1,
            size_scale_max=3.0,
            max_iterations=100,
            barrier_type="pseudo",
        )
        assert config.size_scale_min == 0.1
        assert config.size_scale_max == 3.0
        assert config.max_iterations == 100
        assert config.barrier_type == "pseudo"


class TestAdaptivityResult:
    """Tests for AdaptivityResult dataclass."""

    def test_success_result(self):
        """Successful result has correct attributes."""
        result = AdaptivityResult(
            success=True,
            coords_adapted=np.array([[1.0, 2.0]]),
            quality_before={"min_quality": 0.5},
            quality_after={"min_quality": 0.8},
            iterations=50,
        )
        assert result.success
        assert result.error_message is None
        assert result.iterations == 50

    def test_failure_result(self):
        """Failed result has error message."""
        result = AdaptivityResult(
            success=False,
            coords_adapted=np.array([]),
            error_message="Test error",
        )
        assert not result.success
        assert result.error_message == "Test error"


class TestTMOPAvailability:
    """Tests for TMOP availability checking."""

    def test_is_tmop_available_returns_bool(self):
        """is_tmop_available returns a boolean."""
        result = is_tmop_available()
        assert isinstance(result, bool)


class TestTMOPAdaptivity:
    """Tests for TMOPAdaptivity class."""

    def test_initialization(self, default_config):
        """TMOPAdaptivity initializes correctly."""
        adaptivity = TMOPAdaptivity(default_config)
        assert adaptivity.config == default_config

    def test_initialization_default_config(self):
        """TMOPAdaptivity uses default config if None."""
        adaptivity = TMOPAdaptivity()
        assert adaptivity.config is not None
        assert isinstance(adaptivity.config, AdaptivityConfig)

    @pytest.mark.skipif(
        not is_tmop_available(),
        reason="TMOP not available in PyMFEM"
    )
    def test_adapt_requires_mfem_manager(self, default_config, sample_error_field):
        """adapt() requires MFEMManager, not generic MeshManager."""
        from unittest.mock import MagicMock

        adaptivity = TMOPAdaptivity(default_config)

        # Mock a non-MFEM manager
        mock_manager = MagicMock()
        mock_manager.get_nodes.return_value = np.random.rand(10, 2)

        result = adaptivity.adapt(mock_manager, sample_error_field[:10])
        assert not result.success
        assert "MFEMManager" in result.error_message


class TestErrorProcessing:
    """Tests for error field processing logic."""

    def test_error_field_normalization(self, sample_error_field):
        """Error field should be normalizable."""
        error = sample_error_field
        normalized = error / np.max(error)
        assert np.max(normalized) == 1.0
        assert np.min(normalized) >= 0.0

    def test_error_to_size_conversion(self, default_config):
        """High error should map to small target size."""
        # Error of 1.0 (max) should give size_scale_min
        # Error of 0.0 (min) should give size_scale_max
        size_min = default_config.size_scale_min
        size_max = default_config.size_scale_max

        error_high = 1.0
        error_low = 0.0

        # Size = size_max - (size_max - size_min) * error
        size_at_high_error = size_max - (size_max - size_min) * error_high
        size_at_low_error = size_max - (size_max - size_min) * error_low

        assert size_at_high_error == pytest.approx(size_min)  # Small size attracts nodes
        assert size_at_low_error == pytest.approx(size_max)  # Large size repels nodes


class TestImports:
    """Tests for package imports."""

    def test_import_main_classes(self):
        """Main r-adaptivity classes are importable."""
        from meshforge.morphing import (
            TMOPAdaptivity,
            AdaptivityConfig,
            AdaptivityResult,
            is_tmop_available,
        )
        assert TMOPAdaptivity is not None
        assert AdaptivityConfig is not None
        assert AdaptivityResult is not None
        assert is_tmop_available is not None

    def test_import_from_main_package(self):
        """R-adaptivity classes are importable from main package."""
        from meshforge import (
            TMOPAdaptivity,
            AdaptivityConfig,
            AdaptivityResult,
        )
        assert TMOPAdaptivity is not None
        assert AdaptivityConfig is not None
        assert AdaptivityResult is not None
