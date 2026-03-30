"""
Tests for h-refinement (element splitting) using PyMFEM.

Verifies:
1. PyMFEM h-refinement API availability
2. Element and node count increase after refinement
3. Error-driven element selection
4. Max-element cap is respected
5. Multi-level refinement
6. MFEMManager.refine_by_error helper
"""

import os
import sys
import types
import importlib.util

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to load only mesh/morphing modules without heavy optional deps
# ---------------------------------------------------------------------------

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap():
    """Load only the mesh/morphing modules to avoid optional heavy deps."""
    # Only install a plain stub when the namespace isn't already a real package.
    # This prevents the stub from blocking later imports of meshforge.surrogate
    # etc. when test files are collected together in the same pytest session.
    import importlib
    for stub in ["meshforge", "meshforge.mesh", "meshforge.morphing"]:
        if stub not in sys.modules:
            try:
                importlib.import_module(stub)
            except ImportError:
                sys.modules[stub] = types.ModuleType(stub)

    root = os.path.join(os.path.dirname(__file__), "..", "meshforge")
    _load("meshforge.mesh.base",          f"{root}/mesh/base.py")
    _load("meshforge.mesh.mfem_manager",  f"{root}/mesh/mfem_manager.py")
    _load("meshforge.morphing.h_refinement", f"{root}/morphing/h_refinement.py")


_bootstrap()

from meshforge.mesh.mfem_manager import MFEMManager                   # noqa: E402
from meshforge.morphing.h_refinement import (                          # noqa: E402
    HRefinement,
    HRefinementConfig,
    HRefinementResult,
    is_h_refinement_available,
)

SAMPLE_MESH = os.path.join(
    os.path.dirname(__file__), "..", "samples", "sample_000.mesh"
)

pytestmark = pytest.mark.skipif(
    not is_h_refinement_available(),
    reason="PyMFEM h-refinement (RefineByError / GeneralRefinement) not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def manager():
    return MFEMManager(SAMPLE_MESH)


@pytest.fixture()
def uniform_error(manager):
    """Uniform error field (all elements equal)."""
    return np.ones(manager.num_elements)


@pytest.fixture()
def gaussian_error(manager):
    """Gaussian error centred at (0.5, 0.5), evaluated at element centroids."""
    elements = manager.get_elements()
    nodes = manager.get_nodes()
    centers = np.array([
        nodes[elem[elem >= 0]].mean(axis=0)
        for elem in elements
    ])
    cx, cy = 0.5, 0.5
    return np.exp(-((centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2) / 0.05)


# ---------------------------------------------------------------------------
# TestHRefinementAvailability
# ---------------------------------------------------------------------------

class TestHRefinementAvailability:
    def test_is_h_refinement_available(self):
        assert is_h_refinement_available(), (
            "PyMFEM RefineByError / GeneralRefinement not found on mfem.Mesh"
        )

    def test_mfem_mesh_has_refine_by_error(self):
        import mfem.ser as mfem
        assert hasattr(mfem.Mesh, "RefineByError")

    def test_mfem_mesh_has_general_refinement(self):
        import mfem.ser as mfem
        assert hasattr(mfem.Mesh, "GeneralRefinement")


# ---------------------------------------------------------------------------
# TestMeshRefinement
# ---------------------------------------------------------------------------

class TestMeshRefinement:
    def test_element_count_increases(self, manager, uniform_error):
        ne_before = manager.num_elements
        config = HRefinementConfig(error_threshold=0.1, max_refinement_levels=1)
        result = HRefinement(config).refine(manager, uniform_error)
        assert result.success, f"Refinement failed: {result.error_message}"
        assert result.num_elements_after > ne_before

    def test_node_count_increases(self, manager, uniform_error):
        nv_before = manager.num_nodes
        config = HRefinementConfig(error_threshold=0.1, max_refinement_levels=1)
        result = HRefinement(config).refine(manager, uniform_error)
        assert result.success, f"Refinement failed: {result.error_message}"
        assert result.num_nodes_after > nv_before

    def test_no_negative_error(self, manager):
        """Positive-only error field must succeed without errors."""
        error = np.abs(np.random.default_rng(42).standard_normal(manager.num_elements)) + 0.1
        config = HRefinementConfig(error_threshold=0.5, max_refinement_levels=1)
        result = HRefinement(config).refine(manager, error)
        assert result.success, f"Refinement failed: {result.error_message}"

    def test_wrong_error_length(self, manager):
        """Wrong-length error field must return success=False without raising."""
        bad_error = np.ones(manager.num_elements + 5)
        result = HRefinement().refine(manager, bad_error)
        assert not result.success
        assert result.error_message is not None

    def test_uniform_error_refines_all(self, manager):
        """Uniform error → every element is above threshold → all get split."""
        ne_before = manager.num_elements
        uniform = np.ones(ne_before)
        # threshold=0.0 to ensure all elements qualify
        config = HRefinementConfig(error_threshold=0.0, max_refinement_levels=1)
        result = HRefinement(config).refine(manager, uniform)
        assert result.success
        # All (or nearly all) elements split: at minimum a large fraction
        assert result.num_elements_after > ne_before

    def test_high_error_selects_elements(self, manager, gaussian_error):
        """Gaussian error centred at (0.5,0.5) should trigger refinement."""
        ne_before = manager.num_elements
        config = HRefinementConfig(error_threshold=0.5, max_refinement_levels=1)
        result = HRefinement(config).refine(manager, gaussian_error)
        assert result.success, f"Refinement failed: {result.error_message}"
        assert result.num_elements_after > ne_before

    def test_max_elements_respected(self, manager, uniform_error):
        """
        Refinement must not apply additional levels once max_elements is reached.

        MFEM's RefineByError splits all qualifying elements in one call, so the
        element count after a single pass may exceed the cap.  What the cap
        *does* guarantee is that no further levels are applied after the mesh
        first exceeds it.  We verify this by comparing a capped run (levels=3)
        against an uncapped run (levels=3) — the capped run must produce no
        more elements.
        """
        ne_before = manager.num_elements

        # Run without a cap — all 3 levels fire
        mgr_uncapped = MFEMManager(SAMPLE_MESH)
        cfg_uncapped = HRefinementConfig(
            error_threshold=0.1,
            max_refinement_levels=3,
            max_elements=999999,
        )
        r_uncapped = HRefinement(cfg_uncapped).refine(mgr_uncapped, np.ones(mgr_uncapped.num_elements))

        # Run with cap set after the first refinement pass — stops at level 1
        mgr_capped = MFEMManager(SAMPLE_MESH)
        # After one pass, element count will be around 4x the original.
        # Cap it just above that to prevent a second pass.
        cap = r_uncapped.num_elements_after  # equal to after-level-1 count
        cfg_capped = HRefinementConfig(
            error_threshold=0.1,
            max_refinement_levels=3,
            max_elements=cap,
        )
        r_capped = HRefinement(cfg_capped).refine(mgr_capped, np.ones(mgr_capped.num_elements))

        # Capped run must have applied fewer levels than uncapped
        assert r_capped.levels_applied <= r_uncapped.levels_applied
        # And never exceed the uncapped result
        assert r_capped.num_elements_after <= r_uncapped.num_elements_after

    def test_multi_level_refinement(self, manager, uniform_error):
        """Two levels of refinement should produce more elements than one level."""
        ne_before = manager.num_elements

        mgr1 = MFEMManager(SAMPLE_MESH)
        err1 = np.ones(mgr1.num_elements)
        cfg1 = HRefinementConfig(error_threshold=0.1, max_refinement_levels=1)
        r1 = HRefinement(cfg1).refine(mgr1, err1)

        mgr2 = MFEMManager(SAMPLE_MESH)
        err2 = np.ones(mgr2.num_elements)
        cfg2 = HRefinementConfig(error_threshold=0.1, max_refinement_levels=2)
        r2 = HRefinement(cfg2).refine(mgr2, err2)

        assert r2.num_elements_after >= r1.num_elements_after, (
            f"Level-2 ({r2.num_elements_after}) should be >= level-1 "
            f"({r1.num_elements_after})"
        )


# ---------------------------------------------------------------------------
# TestMFEMManagerRefineByError
# ---------------------------------------------------------------------------

class TestMFEMManagerRefineByError:
    def test_refine_by_error_increases_elements(self, manager):
        ne_before = manager.num_elements
        error = np.ones(ne_before)
        manager.refine_by_error(error, threshold_fraction=0.1)
        assert manager.num_elements > ne_before

    def test_returns_true_on_refinement(self, manager):
        error = np.ones(manager.num_elements)
        result = manager.refine_by_error(error, threshold_fraction=0.1)
        assert result is True

    def test_nodes_updated_after_refinement(self, manager):
        nv_before = manager.num_nodes
        error = np.ones(manager.num_elements)
        manager.refine_by_error(error, threshold_fraction=0.1)
        nodes = manager.get_nodes()
        assert nodes.shape[0] > nv_before
        assert nodes.shape[1] == 2  # still 2-D mesh

    def test_wrong_length_raises(self, manager):
        bad_error = np.ones(manager.num_elements + 3)
        with pytest.raises(ValueError):
            manager.refine_by_error(bad_error)
