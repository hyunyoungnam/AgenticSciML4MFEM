"""
Integration tests for full mesh round-trip cycle.

This module tests the complete workflow:
1. Load a base .mesh file
2. Modify node coordinates
3. Export to new file
4. Re-load and verify modifications are correct
5. Verify unmodified data remains identical
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import will fail if PyMFEM is not installed - skip tests gracefully
try:
    from meshforge.mesh.mfem_manager import MFEMManager
    from meshforge.mesh.base import MeshManager
    from meshforge.evaluation.pipeline import EvaluationPipeline
    from meshforge.evaluation.preflight import PreflightChecker
    from meshforge.evaluation.metrics import MetricsCalculator
    MFEM_AVAILABLE = True
except ImportError:
    MFEM_AVAILABLE = False
    MFEMManager = None


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestMeshRoundTripIntegration:
    """Test cases for complete round-trip cycle with MFEM meshes."""

    def test_full_round_trip_node_modification(self, beam_quad_mesh_file, tmp_output_mesh):
        """
        Complete round-trip: Load -> Modify Nodes -> Save -> Re-load -> Verify.

        This test ensures:
        1. Modified node coordinates are correctly written and read back
        2. Unmodified nodes remain identical
        3. All other data (elements, boundaries) remain unchanged
        """
        # Step 1: Load base mesh file
        manager = MFEMManager(str(beam_quad_mesh_file))

        # Get original data for comparison
        original_nodes = manager.get_nodes().copy()
        original_node_ids = manager.get_node_ids().copy()
        original_elements = manager.get_elements().copy()
        original_element_ids = manager.get_element_ids().copy()
        original_num_nodes = manager.num_nodes
        original_num_elements = manager.num_elements

        # Step 2: Modify a subset of node coordinates
        # Scale first 3 nodes by 1.5
        modified_indices = [0, 1, 2]
        modified_coords = original_nodes[modified_indices] * 1.5

        manager.update_nodes(modified_coords, node_ids=np.array(modified_indices))

        # Verify modification in memory
        assert manager.is_modified, "Mesh should be marked as modified"

        # Step 3: Export modified mesh to new file
        output_path = manager.save(tmp_output_mesh)
        assert Path(output_path).exists(), "Output file should be created"

        # Step 4: Re-load the newly created file
        new_manager = MFEMManager(str(output_path))

        # Step 5: Verify modified values are exactly as expected
        new_nodes = new_manager.get_nodes()
        new_node_ids = new_manager.get_node_ids()

        # Check that modified nodes are correct
        for i, idx in enumerate(modified_indices):
            expected_coords = modified_coords[i]
            actual_coords = new_nodes[idx]
            assert np.allclose(actual_coords, expected_coords, rtol=1e-6), \
                f"Node {idx} coordinates mismatch: expected {expected_coords}, got {actual_coords}"

        # Verify unmodified nodes remain identical
        unmodified_indices = [i for i in range(original_num_nodes) if i not in modified_indices]
        for idx in unmodified_indices:
            orig_coords = original_nodes[idx]
            new_coords = new_nodes[idx]
            assert np.allclose(orig_coords, new_coords, rtol=1e-6), \
                f"Unmodified node {idx} coordinates changed: {orig_coords} vs {new_coords}"

        # Step 6: Verify topology unchanged
        assert new_manager.num_nodes == original_num_nodes, "Node count should remain unchanged"
        assert new_manager.num_elements == original_num_elements, "Element count should remain unchanged"

    def test_full_round_trip_scale_all_nodes(self, beam_quad_mesh_file, tmp_output_mesh):
        """Test scaling all nodes and verifying round-trip."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        original_nodes = manager.get_nodes().copy()
        scale_factor = 2.0

        # Scale all nodes
        new_nodes = original_nodes * scale_factor
        manager.update_nodes(new_nodes)

        # Save and reload
        output_path = manager.save(tmp_output_mesh)
        new_manager = MFEMManager(str(output_path))

        # Verify scaling
        loaded_nodes = new_manager.get_nodes()
        assert np.allclose(loaded_nodes, new_nodes, rtol=1e-6), \
            "Scaled nodes should match after round-trip"

    def test_round_trip_preserves_element_topology(self, beam_quad_mesh_file, tmp_output_mesh):
        """Test that element connectivity is preserved through round-trip."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        original_elements = manager.get_elements().copy()
        original_element_ids = manager.get_element_ids().copy()

        # Modify nodes (but not elements)
        nodes = manager.get_nodes()
        manager.update_nodes(nodes * 1.1)

        # Save and reload
        output_path = manager.save(tmp_output_mesh)
        new_manager = MFEMManager(str(output_path))

        # Verify element topology unchanged
        new_elements = new_manager.get_elements()
        new_element_ids = new_manager.get_element_ids()

        assert np.array_equal(original_element_ids, new_element_ids), \
            "Element IDs should remain unchanged"
        assert np.array_equal(original_elements, new_elements), \
            "Element connectivity should remain unchanged"


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestMeshDimensionHandling:
    """Test cases for 2D and 3D mesh handling."""

    def test_2d_mesh_round_trip(self, beam_quad_mesh_file, tmp_output_mesh):
        """Test 2D mesh round-trip."""
        manager = MFEMManager(str(beam_quad_mesh_file))
        assert manager.dimension == 2, "beam-quad should be 2D"

        nodes = manager.get_nodes()
        assert nodes.shape[1] == 2, "2D nodes should have 2 coordinates"

        # Modify and save
        manager.update_nodes(nodes * 1.2)
        output_path = manager.save(tmp_output_mesh)

        # Reload and verify
        new_manager = MFEMManager(str(output_path))
        assert new_manager.dimension == 2, "Dimension should be preserved"

    def test_3d_mesh_round_trip(self, beam_hex_mesh_file, tmp_path):
        """Test 3D mesh round-trip."""
        manager = MFEMManager(str(beam_hex_mesh_file))
        assert manager.dimension == 3, "beam-hex should be 3D"

        nodes = manager.get_nodes()
        assert nodes.shape[1] == 3, "3D nodes should have 3 coordinates"

        # Modify and save
        manager.update_nodes(nodes * 1.2)
        output_path = tmp_path / "output_3d.mesh"
        manager.save(output_path)

        # Reload and verify
        new_manager = MFEMManager(str(output_path))
        assert new_manager.dimension == 3, "Dimension should be preserved"


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestEvaluationPipelineIntegration:
    """Test cases for integration with evaluation pipeline."""

    def test_evaluation_pipeline_with_mesh(self, beam_quad_mesh_file):
        """Test evaluation pipeline with MFEM mesh."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        # Create evaluation pipeline
        pipeline = EvaluationPipeline(
            preflight_checker=PreflightChecker(),
            metrics_calculator=MetricsCalculator(),
            run_solver=False,  # Skip solver for basic test
        )

        # Run evaluation
        result = pipeline.evaluate(manager)

        # Verify evaluation succeeded
        assert result is not None
        assert result.preflight_passed, "Preflight should pass for valid mesh"
        assert result.overall_score > 0, "Should have positive score"

    def test_evaluation_pipeline_with_modified_mesh(self, beam_quad_mesh_file):
        """Test evaluation pipeline after mesh modification."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        # Modify mesh
        nodes = manager.get_nodes()
        manager.update_nodes(nodes * 1.1)

        # Create evaluation pipeline
        pipeline = EvaluationPipeline(
            preflight_checker=PreflightChecker(),
            metrics_calculator=MetricsCalculator(),
            run_solver=False,
        )

        # Run evaluation
        result = pipeline.evaluate(manager)

        # Verification should still work on modified mesh
        assert result is not None
        assert result.preflight_passed, "Preflight should pass for scaled mesh"


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestDataIntegrity:
    """Test cases for data integrity during operations."""

    def test_no_data_loss_during_round_trip(self, beam_quad_mesh_file, tmp_output_mesh):
        """Verify that no data is lost during round-trip."""
        # Load original
        original_manager = MFEMManager(str(beam_quad_mesh_file))

        # Count original entities
        original_num_nodes = original_manager.num_nodes
        original_num_elements = original_manager.num_elements
        original_dimension = original_manager.dimension

        # Make a modification
        nodes = original_manager.get_nodes()
        original_manager.update_nodes(nodes * 1.01)

        # Save
        output_path = original_manager.save(tmp_output_mesh)

        # Re-load
        new_manager = MFEMManager(str(output_path))

        # Verify counts match
        assert new_manager.num_nodes == original_num_nodes, \
            f"Node count mismatch: {original_num_nodes} vs {new_manager.num_nodes}"
        assert new_manager.num_elements == original_num_elements, \
            f"Element count mismatch: {original_num_elements} vs {new_manager.num_elements}"
        assert new_manager.dimension == original_dimension, \
            f"Dimension mismatch: {original_dimension} vs {new_manager.dimension}"

    def test_bounding_box_changes_with_scaling(self, beam_quad_mesh_file):
        """Test that bounding box correctly reflects node scaling."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        min_orig, max_orig = manager.get_bounding_box()
        scale_factor = 2.0

        # Scale all nodes
        nodes = manager.get_nodes()
        manager.update_nodes(nodes * scale_factor)

        min_new, max_new = manager.get_bounding_box()

        # Bounding box should scale proportionally
        assert np.allclose(min_new, min_orig * scale_factor, rtol=1e-6), \
            "Min bounding box should scale"
        assert np.allclose(max_new, max_orig * scale_factor, rtol=1e-6), \
            "Max bounding box should scale"

    def test_centroid_changes_with_translation(self, beam_quad_mesh_file):
        """Test that centroid correctly reflects node translation."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        centroid_orig = manager.get_centroid()
        translation = np.array([5.0, 10.0])

        # Translate all nodes
        nodes = manager.get_nodes()
        manager.update_nodes(nodes + translation)

        centroid_new = manager.get_centroid()

        # Centroid should translate by the same amount
        assert np.allclose(centroid_new, centroid_orig + translation, rtol=1e-6), \
            "Centroid should translate with nodes"


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestMeshRefinementIntegration:
    """Test cases for mesh refinement workflow."""

    def test_refine_and_save(self, beam_quad_mesh_file, tmp_path):
        """Test refining mesh and saving result."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        original_num_nodes = manager.num_nodes
        original_num_elements = manager.num_elements

        # Refine
        manager.refine_uniformly(times=1)

        assert manager.num_nodes > original_num_nodes, "Should have more nodes after refinement"
        assert manager.num_elements > original_num_elements, "Should have more elements after refinement"

        # Save refined mesh
        output_path = tmp_path / "refined.mesh"
        manager.save(output_path)

        # Reload and verify
        refined_manager = MFEMManager(str(output_path))
        assert refined_manager.num_nodes == manager.num_nodes
        assert refined_manager.num_elements == manager.num_elements
