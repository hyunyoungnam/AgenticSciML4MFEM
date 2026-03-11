"""
Tests for morphing.py: region-based morphing and IDW.

Tests the core morphing functions that operate on coordinate arrays,
independent of the specific mesh manager implementation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from meshforge.morphing import (
    load_morphing_config,
    assign_regions_from_geometry,
    compute_moving_displacements,
    idw_displacements,
    MorphingContext,
    ROLE_MOVING,
    ROLE_ANCHOR,
    ROLE_MORPHING,
)


@pytest.fixture
def project_root():
    """Project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def morphing_config_path(project_root):
    """Path to the quarter plate with hole morphing config .md."""
    config_file = project_root / "meshforge" / "configs" / "quarter_plate_with_hole_morphing.md"
    if not config_file.exists():
        pytest.skip(f"Morphing config not found at {config_file}")
    return config_file


@pytest.fixture
def sample_2d_coords():
    """Sample 2D coordinates for testing morphing."""
    # Create a simple grid with a hole at origin
    coords = []
    for r in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        for theta in np.linspace(0, np.pi/2, 5):
            coords.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(coords)


@pytest.fixture
def sample_morphing_config():
    """Sample morphing configuration for testing."""
    return {
        "_R0": 1.0,
        "_R_transition": 5.0,
        "_hole_center": np.array([0.0, 0.0]),
        "_tolerance": 0.1,
        "regions": {
            "hole_boundary": {"role": "moving", "idw_p": 4},
            "transition": {"role": "morphing", "idw_p": 2},
            "far_field": {"role": "anchor", "idw_p": None},
        }
    }


class TestMorphingConfig:
    """Tests for loading the morphing config from markdown."""

    def test_load_morphing_config_returns_geometry_and_regions(
        self, morphing_config_path
    ):
        """Config load returns geometry (R0, R_transition, hole_center) and regions."""
        config = load_morphing_config(morphing_config_path)
        assert "_R0" in config
        assert "_R_transition" in config
        assert "_hole_center" in config
        assert "_tolerance" in config
        assert "regions" in config
        assert "hole_boundary" in config["regions"]
        assert "transition" in config["regions"]
        assert "far_field" in config["regions"]

    def test_region_roles_and_idw_p(self, morphing_config_path):
        """Regions have expected roles and idw_p."""
        config = load_morphing_config(morphing_config_path)
        regions = config["regions"]
        assert regions["hole_boundary"]["role"] == "moving"
        assert regions["transition"]["role"] == "morphing"
        assert regions["far_field"]["role"] == "anchor"
        assert regions["hole_boundary"].get("idw_p") is not None or regions["transition"].get("idw_p") is not None


class TestRegionAssignment:
    """Tests for region assignment from geometry."""

    def test_assign_regions_from_geometry_produces_three_roles(
        self, sample_2d_coords, sample_morphing_config
    ):
        """Assignment yields moving, anchor, and morphing nodes."""
        region_id, roles = assign_regions_from_geometry(
            sample_2d_coords, sample_morphing_config
        )

        # Should have all three roles
        assert np.any(roles == ROLE_MOVING), "Should have moving nodes"
        assert np.any(roles == ROLE_ANCHOR), "Should have anchor nodes"
        assert np.any(roles == ROLE_MORPHING), "Should have morphing nodes"

        # Lengths should match
        assert len(region_id) == len(sample_2d_coords)
        assert len(roles) == len(sample_2d_coords)

    def test_moving_nodes_near_hole(self, sample_2d_coords, sample_morphing_config):
        """Moving nodes should be near the hole (at R0)."""
        region_id, roles = assign_regions_from_geometry(
            sample_2d_coords, sample_morphing_config
        )

        R0 = sample_morphing_config["_R0"]
        tol = sample_morphing_config["_tolerance"]
        center = sample_morphing_config["_hole_center"]

        # Compute distances from center
        distances = np.linalg.norm(sample_2d_coords - center, axis=1)

        # Moving nodes should be near R0
        moving_mask = roles == ROLE_MOVING
        if np.any(moving_mask):
            moving_distances = distances[moving_mask]
            assert np.all(moving_distances <= R0 + tol + 0.5), \
                "Moving nodes should be near hole boundary"

    def test_anchor_nodes_far_from_hole(self, sample_2d_coords, sample_morphing_config):
        """Anchor nodes should be far from the hole (beyond R_transition)."""
        region_id, roles = assign_regions_from_geometry(
            sample_2d_coords, sample_morphing_config
        )

        R_transition = sample_morphing_config["_R_transition"]
        center = sample_morphing_config["_hole_center"]

        # Compute distances from center
        distances = np.linalg.norm(sample_2d_coords - center, axis=1)

        # Anchor nodes should be beyond transition
        anchor_mask = roles == ROLE_ANCHOR
        if np.any(anchor_mask):
            anchor_distances = distances[anchor_mask]
            assert np.all(anchor_distances >= R_transition - 0.5), \
                "Anchor nodes should be beyond transition region"


class TestDisplacementComputation:
    """Tests for displacement computation functions."""

    def test_compute_moving_displacements_radial(self):
        """Moving displacements should be radial outward."""
        # Create circular nodes at radius 2.0
        theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
        coords = np.column_stack([2.0 * np.cos(theta), 2.0 * np.sin(theta)])
        center = np.array([0.0, 0.0])
        delta_R = 0.5

        displacements = compute_moving_displacements(coords, center, delta_R)

        # Displacements should be radial (parallel to position vector)
        for i in range(len(coords)):
            pos = coords[i] - center
            disp = displacements[i]

            # Normalize both vectors
            pos_norm = pos / np.linalg.norm(pos)
            disp_norm = disp / np.linalg.norm(disp)

            # Should be parallel (dot product = 1)
            assert np.allclose(np.abs(np.dot(pos_norm, disp_norm)), 1.0, atol=0.01), \
                "Displacement should be radial"

    def test_compute_moving_displacements_magnitude(self):
        """Moving displacements should have magnitude equal to delta_R."""
        coords = np.array([[2.0, 0.0], [0.0, 2.0], [1.414, 1.414]])
        center = np.array([0.0, 0.0])
        delta_R = 0.3

        displacements = compute_moving_displacements(coords, center, delta_R)

        magnitudes = np.linalg.norm(displacements, axis=1)
        assert np.allclose(magnitudes, delta_R, atol=0.01), \
            f"Displacement magnitudes should be {delta_R}, got {magnitudes}"


class TestIDWInterpolation:
    """Tests for IDW displacement interpolation."""

    def test_idw_displacements_anchor_zero(self):
        """Anchor nodes should have zero displacement after IDW."""
        # Setup: moving nodes at radius 2, anchor nodes at radius 10
        moving_coords = np.array([[2.0, 0.0], [0.0, 2.0]])
        anchor_coords = np.array([[10.0, 0.0], [0.0, 10.0]])
        morphing_coords = np.array([[5.0, 0.0], [0.0, 5.0]])

        all_coords = np.vstack([moving_coords, anchor_coords, morphing_coords])
        roles = np.array([ROLE_MOVING, ROLE_MOVING, ROLE_ANCHOR, ROLE_ANCHOR,
                          ROLE_MORPHING, ROLE_MORPHING])

        # Moving displacements
        center = np.array([0.0, 0.0])
        moving_disps = np.array([[0.5, 0.0], [0.0, 0.5]])  # Radial outward
        anchor_disps = np.array([[0.0, 0.0], [0.0, 0.0]])  # Fixed

        known_disps = np.vstack([moving_disps, anchor_disps])
        known_mask = (roles == ROLE_MOVING) | (roles == ROLE_ANCHOR)
        morphing_mask = roles == ROLE_MORPHING

        idw_p = 2.0
        morphing_disps = idw_displacements(
            all_coords,
            known_mask,
            morphing_mask,
            known_disps,
            p=idw_p,
        )

        # Morphing nodes should have interpolated displacement
        assert morphing_disps.shape[0] == np.sum(morphing_mask)
        assert np.all(np.linalg.norm(morphing_disps, axis=1) > 0), \
            "Morphing nodes should have non-zero displacement"
        assert np.all(np.linalg.norm(morphing_disps, axis=1) < 0.5), \
            "Morphing nodes should have smaller displacement than moving nodes"

    def test_idw_displacements_decay_with_distance(self):
        """IDW displacement should decay with distance from moving nodes."""
        # Setup: moving node at origin, morphing nodes at increasing distances
        moving_coords = np.array([[0.0, 0.0]])
        morphing_coords = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
        anchor_coords = np.array([[10.0, 0.0]])

        all_coords = np.vstack([moving_coords, morphing_coords, anchor_coords])
        roles = np.array([ROLE_MOVING, ROLE_MORPHING, ROLE_MORPHING, ROLE_MORPHING, ROLE_ANCHOR])

        moving_disps = np.array([[1.0, 0.0]])
        anchor_disps = np.array([[0.0, 0.0]])
        known_disps = np.vstack([moving_disps, anchor_disps])

        known_mask = (roles == ROLE_MOVING) | (roles == ROLE_ANCHOR)
        morphing_mask = roles == ROLE_MORPHING

        morphing_disps = idw_displacements(
            all_coords, known_mask, morphing_mask, known_disps, p=2.0
        )

        # Displacement should decrease with distance
        disp_mags = np.linalg.norm(morphing_disps, axis=1)
        assert disp_mags[0] > disp_mags[1] > disp_mags[2], \
            "Displacement should decay with distance from moving node"


class TestMorphingContext:
    """Tests for MorphingContext dataclass."""

    def test_morphing_context_masks(self):
        """Test mask methods of MorphingContext."""
        n = 10
        ctx = MorphingContext(
            node_ids=np.arange(n),
            coords_original=np.random.rand(n, 2),
            coords_morphed=np.random.rand(n, 2),
            role=np.array([ROLE_MOVING, ROLE_MOVING, ROLE_ANCHOR, ROLE_ANCHOR,
                           ROLE_MORPHING, ROLE_MORPHING, ROLE_MORPHING,
                           ROLE_ANCHOR, ROLE_MOVING, ROLE_ANCHOR]),
            distance_from_center=np.random.rand(n),
            displacement=np.random.rand(n, 2),
        )

        moving_mask = ctx.get_moving_mask()
        anchor_mask = ctx.get_anchor_mask()
        morphing_mask = ctx.get_morphing_mask()

        assert np.sum(moving_mask) == 3
        assert np.sum(anchor_mask) == 4
        assert np.sum(morphing_mask) == 3

        # Masks should be mutually exclusive
        assert np.sum(moving_mask & anchor_mask) == 0
        assert np.sum(moving_mask & morphing_mask) == 0
        assert np.sum(anchor_mask & morphing_mask) == 0

    def test_morphing_context_summary(self):
        """Test summary method of MorphingContext."""
        n = 5
        ctx = MorphingContext(
            node_ids=np.arange(n),
            coords_original=np.zeros((n, 2)),
            coords_morphed=np.zeros((n, 2)),
            role=np.array([ROLE_MOVING, ROLE_ANCHOR, ROLE_MORPHING, ROLE_MOVING, ROLE_ANCHOR]),
            distance_from_center=np.zeros(n),
            displacement=np.random.rand(n, 2),
            delta_R=0.5,
            R0=2.0,
            R_target=2.5,
            R_transition=5.0,
        )

        summary = ctx.summary()

        assert summary["total_nodes"] == n
        assert summary["moving_count"] == 2
        assert summary["anchor_count"] == 2
        assert summary["morphing_count"] == 1
        assert summary["delta_R"] == 0.5
        assert summary["R0"] == 2.0
        assert summary["R_target"] == 2.5
        assert "max_displacement" in summary
