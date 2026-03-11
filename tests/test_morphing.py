"""
Tests for morphing.py: region-based morphing and IDW.

Tests the quarter plate with hole example: slightly increase hole radius
via morphing config, then write the morphed mesh to a new .inp file.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from inpforge.manager import AbaqusManager
from inpforge.morphing import (
    load_morphing_config,
    run_morphing,
    morph_and_write,
    assign_regions_from_geometry,
)
from inpforge.parser import AbaqusParser


@pytest.fixture
def project_root():
    """Project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def base_inp_file(project_root):
    """Path to the quarter plate with hole base .inp file."""
    inp_file = project_root / "inputs" / "BaseInp2D.inp"
    assert inp_file.exists(), f"BaseInp2D.inp not found at {inp_file}"
    return inp_file


@pytest.fixture
def morphing_config_path(project_root):
    """Path to the quarter plate with hole morphing config .md."""
    config_file = project_root / "configs" / "quarter_plate_with_hole_morphing.md"
    assert config_file.exists(), f"Morphing config not found at {config_file}"
    return config_file


@pytest.fixture
def morphed_output_path(project_root):
    """Output path for morphed .inp (outputs dir); uses standard name OutputInp2D_morphed.inp."""
    from writer import OUTPUT_MORPHED_INP
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    return outputs_dir / OUTPUT_MORPHED_INP


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
        assert regions["hole_boundary"].get("idw_p") == 4
        assert regions["transition"].get("idw_p") == 2


class TestRegionAssignment:
    """Tests for region assignment from geometry."""

    def test_assign_regions_from_geometry_produces_three_roles(
        self, base_inp_file, morphing_config_path
    ):
        """Assignment yields moving, anchor, and morphing nodes."""
        manager = AbaqusManager(str(base_inp_file))
        config = load_morphing_config(morphing_config_path)
        coords = manager.nodes.data
        region_id, roles = assign_regions_from_geometry(coords, config)
        # roles: 0=moving, 1=anchor, 2=morphing
        assert np.any(roles == 0), "Should have moving nodes"
        assert np.any(roles == 1), "Should have anchor nodes"
        assert np.any(roles == 2), "Should have morphing nodes"
        assert len(region_id) == len(coords)
        assert len(roles) == len(coords)


class TestMorphingQuarterPlateWithHole:
    """Tests for slightly increasing hole radius and writing .inp."""

    def test_run_morphing_updates_coordinates(self, base_inp_file, morphing_config_path):
        """run_morphing updates manager node coordinates; max change ~ delta_R."""
        manager = AbaqusManager(str(base_inp_file))
        coords_before = manager.nodes.data.copy()
        delta_R = 0.15
        run_morphing(manager, morphing_config_path, delta_R=delta_R)
        coords_after = manager.nodes.data
        diff = np.abs(coords_after - coords_before)
        assert diff.max() > 0, "Some coordinates should change"
        assert diff.max() <= delta_R * 1.01, "Max change should not exceed delta_R (with small tol)"
        assert manager.nodes.modified
        assert "NODE" in manager.get_modified_sections()

    def test_morph_and_write_creates_valid_inp_file(
        self, base_inp_file, morphing_config_path, morphed_output_path
    ):
        """
        morph_and_write: slightly increase hole radius and write morphed .inp.
        Output file exists, is parseable, has same node/element count, coordinates changed.
        """
        delta_R = 0.1

        inp_out, vtu_out = morph_and_write(
            base_inp_file,
            morphing_config_path,
            morphed_output_path,
            delta_R=delta_R,
            reassign_anchors=True,
        )

        assert Path(inp_out).exists(), "Morphed .inp file should be created"
        assert Path(vtu_out).exists(), "Morphed .vtu file should be created"

        # Re-parse the morphed file
        parser_out = AbaqusParser(inp_out)
        manager_orig = AbaqusManager(str(base_inp_file))
        manager_morphed = AbaqusManager(inp_out)

        # Same topology: node and element counts unchanged
        assert len(manager_morphed.nodes.ids) == len(manager_orig.nodes.ids)
        assert len(manager_morphed.elements.ids) == len(manager_orig.elements.ids)

        # Coordinates changed (morphing applied)
        coords_orig = manager_orig.nodes.data
        coords_morphed = manager_morphed.nodes.data
        diff = np.abs(coords_morphed - coords_orig)
        assert diff.max() > 0, "Morphed coordinates should differ from original"

        # Morphed file is valid (parser found keywords)
        assert len(parser_out.get_all_keywords()) > 0
        assert parser_out.get_keyword_chunks("NODE")
        assert parser_out.get_keyword_chunks("ELEMENT")

    def test_morphed_output_near_hole_moved_more(
        self, base_inp_file, morphing_config_path, morphed_output_path
    ):
        """Nodes near the hole (small distance from origin) move more than far nodes."""
        delta_R = 0.2
        inp_out, vtu_out = morph_and_write(
            base_inp_file,
            morphing_config_path,
            morphed_output_path,
            delta_R=delta_R,
        )

        manager_orig = AbaqusManager(str(base_inp_file))
        manager_morphed = AbaqusManager(inp_out)

        coords_orig = manager_orig.nodes.data
        coords_morphed = manager_morphed.nodes.data
        dist_from_origin_orig = np.sqrt(np.sum(coords_orig ** 2, axis=1))

        # Near hole: distance < 5
        near = dist_from_origin_orig < 5
        # Far: distance > 10
        far = dist_from_origin_orig > 10

        assert np.any(near) and np.any(far)
        change_near = np.abs(coords_morphed[near] - coords_orig[near]).max()
        change_far = np.abs(coords_morphed[far] - coords_orig[far]).max()
        assert change_near >= change_far, (
            "Nodes near the hole should move at least as much as far nodes"
        )
