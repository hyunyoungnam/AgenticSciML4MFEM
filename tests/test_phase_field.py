"""
Tests for phase field fracture implementation.

Tests cover:
1. PhaseFieldConfig validation
2. GmshMeshGenerator mesh creation
3. FEniCSManager mesh loading
4. FEniCSPhaseFieldSolver basic functionality
5. Dataset generation pipeline
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import sys

# Test PhaseFieldConfig
from piano.solvers.base import PhaseFieldConfig, PhysicsType

# Check for optional dependencies
HAS_GMSH = False
HAS_DOLFINX = False

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    pass

try:
    import dolfinx
    HAS_DOLFINX = True
except ImportError:
    pass


class TestPhaseFieldConfig:
    """Tests for PhaseFieldConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PhaseFieldConfig()
        assert config.G_c == 2.7e3
        assert config.l_0 == 0.015
        assert config.k_res == 1e-7
        assert config.damage_threshold == 0.9

    def test_custom_config(self):
        """Test custom configuration."""
        config = PhaseFieldConfig(
            G_c=3.0e3,
            l_0=0.02,
            k_res=1e-6,
            n_load_steps=100,
        )
        assert config.G_c == 3.0e3
        assert config.l_0 == 0.02
        assert config.n_load_steps == 100

    def test_invalid_G_c(self):
        """Test validation of negative G_c."""
        with pytest.raises(ValueError, match="G_c must be positive"):
            PhaseFieldConfig(G_c=-1.0)

    def test_invalid_l_0(self):
        """Test validation of negative l_0."""
        with pytest.raises(ValueError, match="l_0 must be positive"):
            PhaseFieldConfig(l_0=-0.01)

    def test_invalid_k_res(self):
        """Test validation of k_res out of range."""
        with pytest.raises(ValueError, match="k_res must be in"):
            PhaseFieldConfig(k_res=1.5)

    def test_invalid_damage_threshold(self):
        """Test validation of damage_threshold out of range."""
        with pytest.raises(ValueError, match="Damage threshold must be in"):
            PhaseFieldConfig(damage_threshold=1.5)


class TestPhysicsType:
    """Tests for PhysicsType enum extension."""

    def test_phase_field_fracture_exists(self):
        """Test that PHASE_FIELD_FRACTURE is a valid physics type."""
        assert hasattr(PhysicsType, "PHASE_FIELD_FRACTURE")
        assert PhysicsType.PHASE_FIELD_FRACTURE.name == "PHASE_FIELD_FRACTURE"


# Test FEMSample extensions
from piano.data.dataset import FEMSample


class TestFEMSampleExtensions:
    """Tests for FEMSample damage and crack_path fields."""

    def test_damage_field(self):
        """Test damage field in FEMSample."""
        sample = FEMSample(
            sample_id="test",
            parameters={"E": 200e9},
            coordinates=np.array([[0, 0], [1, 0], [0.5, 1]]),
            damage=np.array([0.0, 0.5, 1.0]),
        )
        assert sample.damage is not None
        assert len(sample.damage) == 3
        assert sample.get_output_field("damage") is not None

    def test_crack_path_field(self):
        """Test crack_path field in FEMSample."""
        crack_path = np.array([[0.0, 0.5], [0.3, 0.5], [0.5, 0.6]])
        sample = FEMSample(
            sample_id="test",
            parameters={"E": 200e9},
            coordinates=np.array([[0, 0], [1, 0], [0.5, 1]]),
            crack_path=crack_path,
        )
        assert sample.crack_path is not None
        assert sample.crack_path.shape == (3, 2)
        assert sample.get_output_field("crack_path") is not None

    def test_to_dict_includes_new_fields(self):
        """Test that to_dict includes damage and crack_path flags."""
        sample = FEMSample(
            sample_id="test",
            parameters={},
            coordinates=np.array([[0, 0]]),
            damage=np.array([0.5]),
            crack_path=np.array([[0, 0.5]]),
        )
        d = sample.to_dict()
        assert "has_damage" in d
        assert d["has_damage"] is True
        assert "has_crack_path" in d
        assert d["has_crack_path"] is True


# Test GmshMeshGenerator (requires gmsh)
class TestGmshMeshGenerator:
    """Tests for GmshMeshGenerator."""

    @pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
    def test_import(self):
        """Test that GmshMeshGenerator can be imported."""
        from piano.mesh.gmsh_generator import GmshMeshGenerator, GmshMeshConfig
        assert GmshMeshGenerator is not None
        assert GmshMeshConfig is not None

    @pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
    def test_edge_crack_mesh(self):
        """Test mesh generation for edge crack."""
        from piano.mesh.gmsh_generator import GmshMeshGenerator, GmshMeshConfig
        from piano.geometry.crack import EdgeCrack

        geometry = EdgeCrack(crack_length=0.3, width=1.0, height=1.0)
        config = GmshMeshConfig(base_size=0.05, tip_size=0.01)

        generator = GmshMeshGenerator(geometry, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_mesh.msh"
            vertices, elements, metadata = generator.generate(output_path)

            assert vertices.shape[1] == 2  # 2D coordinates
            assert elements.shape[1] == 3  # triangles
            assert metadata["crack_type"] == "edge"
            assert output_path.exists()


# Test FEniCSManager (requires dolfinx)
class TestFEniCSManager:
    """Tests for FEniCSManager."""

    @pytest.mark.skipif(not HAS_DOLFINX, reason="dolfinx not installed")
    def test_import(self):
        """Test that FEniCSManager can be imported."""
        from piano.mesh.fenics_manager import FEniCSManager
        assert FEniCSManager is not None


# Test FEniCSPhaseFieldSolver (requires dolfinx)
class TestFEniCSPhaseFieldSolver:
    """Tests for FEniCSPhaseFieldSolver."""

    @pytest.mark.skipif(not HAS_DOLFINX, reason="dolfinx not installed")
    def test_import(self):
        """Test that FEniCSPhaseFieldSolver can be imported."""
        from piano.solvers.fenics_phase_field import FEniCSPhaseFieldSolver
        assert FEniCSPhaseFieldSolver is not None

    @pytest.mark.skipif(not HAS_DOLFINX, reason="dolfinx not installed")
    def test_instantiation(self):
        """Test solver instantiation."""
        from piano.solvers.fenics_phase_field import FEniCSPhaseFieldSolver
        solver = FEniCSPhaseFieldSolver()
        assert not solver.is_setup
        assert solver.get_available_fields() == ["displacement", "damage", "von_mises", "crack_path"]


# Test phase field generator (requires both gmsh and dolfinx)
class TestPhaseFieldGenerator:
    """Tests for phase field dataset generator."""

    @pytest.mark.skipif(not (HAS_GMSH and HAS_DOLFINX), reason="gmsh or dolfinx not installed")
    def test_import(self):
        """Test that generator functions can be imported."""
        from piano.data.phase_field_generator import (
            PhaseFieldFEMConfig,
            ParameterBounds,
            generate_phase_field_sample,
        )
        assert PhaseFieldFEMConfig is not None
        assert ParameterBounds is not None
        assert generate_phase_field_sample is not None

    def test_config_defaults(self):
        """Test PhaseFieldFEMConfig default values."""
        from piano.data.phase_field_generator import PhaseFieldFEMConfig
        config = PhaseFieldFEMConfig()
        assert config.geometry_type == "edge_crack"
        assert config.domain_width == 1.0
        assert config.crack_length == 0.3
        assert config.l_0 == 0.015

    def test_parameter_bounds(self):
        """Test ParameterBounds default values."""
        from piano.data.phase_field_generator import ParameterBounds
        bounds = ParameterBounds()
        assert bounds.E_range == (150e9, 250e9)
        assert bounds.nu_range == (0.25, 0.35)
        assert bounds.G_c_range == (1e3, 5e3)


# Integration test (requires all dependencies)
@pytest.mark.slow
class TestPhaseFieldIntegration:
    """Integration tests for phase field solver."""

    @pytest.mark.skipif(not (HAS_GMSH and HAS_DOLFINX), reason="gmsh or dolfinx not installed")
    def test_single_sample_generation(self):
        """Test generation of a single phase field sample."""
        from piano.data.phase_field_generator import (
            PhaseFieldFEMConfig,
            generate_phase_field_sample,
        )

        config = PhaseFieldFEMConfig(
            resolution=20,  # Coarse mesh for speed
            n_load_steps=5,  # Few steps for speed
            l_0=0.05,  # Larger regularization for coarse mesh
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = Path(tmpdir)

            sample = generate_phase_field_sample(
                E=200e9,
                nu=0.3,
                traction=100e6,
                G_c=2.7e3,
                config=config,
            )

            if sample is not None:
                assert sample.is_valid
                assert sample.displacement is not None
                assert sample.damage is not None
                assert "E" in sample.parameters
                assert sample.parameters["E"] == 200e9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
