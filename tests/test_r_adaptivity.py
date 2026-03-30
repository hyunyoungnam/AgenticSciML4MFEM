"""
Tests for TMOP r-adaptivity using PyMFEM.

Verifies:
1. PyMFEM TMOP availability
2. Mesh loading via MFEMManager
3. Node movement under an error-driven field
4. No inverted elements after adaptation
5. Mesh quality improvement
6. Boundary node preservation
7. Save/reload round-trip
"""

import ctypes
import os
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers for importing without triggering torch/deepxde in meshforge.__init__
# ---------------------------------------------------------------------------
import sys
import types
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap():
    """Load only the mesh/morphing modules to avoid optional heavy deps."""
    import importlib
    for stub in ["meshforge", "meshforge.mesh", "meshforge.morphing"]:
        if stub not in sys.modules:
            try:
                importlib.import_module(stub)
            except ImportError:
                sys.modules[stub] = types.ModuleType(stub)

    root = os.path.join(os.path.dirname(__file__), "..", "meshforge")
    _load("meshforge.mesh.base",       f"{root}/mesh/base.py")
    _load("meshforge.mesh.mfem_manager", f"{root}/mesh/mfem_manager.py")
    _load("meshforge.morphing.r_adaptivity", f"{root}/morphing/r_adaptivity.py")


_bootstrap()

from meshforge.mesh.mfem_manager import MFEMManager          # noqa: E402
from meshforge.morphing.r_adaptivity import (                 # noqa: E402
    AdaptivityConfig,
    AdaptivityResult,
    TMOPAdaptivity,
    is_tmop_available,
)

# Path to sample mesh included with the project
SAMPLE_MESH = os.path.join(
    os.path.dirname(__file__), "..", "samples", "sample_000.mesh"
)

pytestmark = pytest.mark.skipif(
    not is_tmop_available(),
    reason="PyMFEM TMOP not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def manager():
    return MFEMManager(SAMPLE_MESH)


@pytest.fixture()
def gaussian_error(manager):
    """Gaussian error field centred at (0.5, 0.5)."""
    coords = manager.get_nodes()
    cx, cy = 0.5, 0.5
    return np.exp(-((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2) / 0.05)


@pytest.fixture()
def config():
    return AdaptivityConfig(max_iterations=100, verbosity=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTMOPAvailability:
    def test_tmop_available(self):
        assert is_tmop_available(), "PyMFEM TMOP classes not found"

    def test_mfem_import(self):
        import mfem.ser as mfem
        assert hasattr(mfem, "tmop")
        assert hasattr(mfem.tmop, "TMOP_Integrator")
        assert hasattr(mfem.tmop, "DiscreteAdaptTC")
        assert hasattr(mfem.tmop, "TMOP_Metric_002")


class TestMeshLoading:
    def test_load_sample_mesh(self, manager):
        assert manager.num_nodes > 0
        assert manager.num_elements > 0
        assert manager.dimension == 2

    def test_node_coordinates_valid(self, manager):
        coords = manager.get_nodes()
        assert coords.ndim == 2
        assert coords.shape[1] == 2
        assert not np.any(np.isnan(coords))
        assert not np.any(np.isinf(coords))

    def test_nodes_within_unit_square(self, manager):
        coords = manager.get_nodes()
        # plate-with-hole mesh lives roughly in [0,1]^2
        assert coords[:, 0].min() >= -1e-10
        assert coords[:, 0].max() <= 1.0 + 1e-10
        assert coords[:, 1].min() >= -1e-10
        assert coords[:, 1].max() <= 1.0 + 1e-10


class TestAdaptivityResult:
    def test_success_flag(self, manager, gaussian_error, config):
        result = TMOPAdaptivity(config).adapt(manager, gaussian_error)
        assert result.success, f"Adaptation failed: {result.error_message}"

    def test_coords_shape(self, manager, gaussian_error, config):
        n = manager.num_nodes
        result = TMOPAdaptivity(config).adapt(manager, gaussian_error)
        assert result.coords_adapted.shape == (n, 2)

    def test_no_nan_or_inf(self, manager, gaussian_error, config):
        result = TMOPAdaptivity(config).adapt(manager, gaussian_error)
        assert not np.any(np.isnan(result.coords_adapted))
        assert not np.any(np.isinf(result.coords_adapted))

    def test_iterations_recorded(self, manager, gaussian_error, config):
        result = TMOPAdaptivity(config).adapt(manager, gaussian_error)
        assert result.iterations > 0


class TestNodeMovement:
    def test_nodes_move(self, manager, gaussian_error, config):
        """At least some interior nodes must relocate."""
        coords_before = manager.get_nodes().copy()
        result = TMOPAdaptivity(config).adapt(manager, gaussian_error)
        disp = np.linalg.norm(result.coords_adapted - coords_before, axis=1)
        assert disp.max() > 1e-6, "No nodes moved — TMOP had no effect"

    def test_boundary_nodes_fixed(self, manager, gaussian_error):
        """With fix_boundary=True, boundary nodes should not move."""
        import mfem.ser as mfem

        coords_before = manager.get_nodes().copy()
        cfg = AdaptivityConfig(max_iterations=50, fix_boundary=True, verbosity=0)
        result = TMOPAdaptivity(cfg).adapt(manager, gaussian_error)
        assert result.success

        # Collect boundary node indices topologically (edges shared by 1 element).
        # mesh.GetNBE() may only annotate a subset of boundary edges; topological
        # detection matches what TMOPAdaptivity._get_boundary_nodes uses.
        from collections import defaultdict
        mesh = manager.mesh
        edge_count = defaultdict(int)
        for e in range(mesh.GetNE()):
            verts = list(mesh.GetElementVertices(e))
            n = len(verts)
            for i in range(n):
                edge = tuple(sorted((verts[i], verts[(i + 1) % n])))
                edge_count[edge] += 1
        bdr_nodes = set()
        for edge, count in edge_count.items():
            if count == 1:
                bdr_nodes.update(edge)

        for idx in bdr_nodes:
            disp = np.linalg.norm(
                result.coords_adapted[idx] - coords_before[idx]
            )
            assert disp < 1e-6, (
                f"Boundary node {idx} moved by {disp:.2e} (should be fixed)"
            )


class TestMeshQuality:
    def test_no_inverted_elements(self, manager, gaussian_error, config):
        result = TMOPAdaptivity(config).adapt(manager, gaussian_error)
        assert result.quality_after["num_inverted"] == 0, (
            f"Inverted elements after adaptation: "
            f"{result.quality_after['num_inverted']}"
        )

    def test_quality_improves(self, manager, gaussian_error, config):
        result = TMOPAdaptivity(config).adapt(manager, gaussian_error)
        q_before = result.quality_before["avg_quality"]
        q_after  = result.quality_after["avg_quality"]
        assert q_after > q_before, (
            f"Quality did not improve: {q_before:.4f} -> {q_after:.4f}"
        )

    def test_quality_metrics_present(self, manager, gaussian_error, config):
        result = TMOPAdaptivity(config).adapt(manager, gaussian_error)
        for key in ("min_quality", "avg_quality", "max_quality", "num_inverted"):
            assert key in result.quality_before
            assert key in result.quality_after


class TestErrorHandling:
    def test_wrong_error_field_length(self, manager, config):
        bad_error = np.ones(manager.num_nodes + 5)
        result = TMOPAdaptivity(config).adapt(manager, bad_error)
        assert not result.success
        assert result.error_message is not None

    def test_uniform_error_field(self, manager, config):
        """Uniform error should still run without crashing."""
        uniform = np.ones(manager.num_nodes) * 0.5
        result = TMOPAdaptivity(config).adapt(manager, uniform)
        assert result.success

    def test_zero_error_field(self, manager, config):
        """All-zero error (no adaptation signal) should not crash."""
        zero = np.zeros(manager.num_nodes)
        result = TMOPAdaptivity(config).adapt(manager, zero)
        assert result.success


class TestSaveReload:
    def test_adapted_mesh_saves_and_reloads(self, manager, gaussian_error, config):
        TMOPAdaptivity(config).adapt(manager, gaussian_error)
        with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as f:
            tmp = f.name
        try:
            saved = manager.save(tmp)
            assert saved.exists()
            mgr2 = MFEMManager(tmp)
            assert mgr2.num_nodes == manager.num_nodes
            assert mgr2.num_elements == manager.num_elements
        finally:
            os.unlink(tmp)

    def test_reloaded_coords_match(self, manager, gaussian_error, config):
        result = TMOPAdaptivity(config).adapt(manager, gaussian_error)
        with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as f:
            tmp = f.name
        try:
            manager.save(tmp)
            mgr2 = MFEMManager(tmp)
            # Coordinates from reloaded mesh should match adapted coords
            np.testing.assert_allclose(
                mgr2.get_nodes(), result.coords_adapted, atol=1e-10
            )
        finally:
            os.unlink(tmp)
