"""
Tests for MFEMManager mesh management.

This module tests the MeshManager interface and MFEMManager implementation:
1. Loading MFEM mesh files
2. Extracting nodes and elements into NumPy arrays
3. Updating node coordinates
4. Saving modified meshes
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
    MFEM_AVAILABLE = True
except ImportError:
    MFEM_AVAILABLE = False
    MFEMManager = None
    MeshManager = None


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestMFEMManagerNodeExtraction:
    """Test cases for node extraction and manipulation."""

    def test_extract_nodes_into_numpy_array(self, beam_quad_mesh_file):
        """Extract nodes into NumPy array and verify shape."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        nodes = manager.get_nodes()
        node_ids = manager.get_node_ids()

        assert nodes is not None, "Nodes should not be None"
        assert node_ids is not None, "Node IDs should not be None"

        # Verify types
        assert isinstance(nodes, np.ndarray), "Nodes should be NumPy array"
        assert isinstance(node_ids, np.ndarray), "Node IDs should be NumPy array"

        # Verify shapes
        assert len(nodes.shape) == 2, "Nodes should be 2D array"
        assert nodes.shape[1] == 2, "beam-quad is 2D mesh"
        assert len(node_ids) == len(nodes), "Node IDs should match node count"

        # Verify data types
        assert nodes.dtype in [np.float64, np.float32], "Coordinates should be float"
        assert node_ids.dtype in [np.int32, np.int64], "Node IDs should be integer"

    def test_extract_nodes_3d(self, beam_hex_mesh_file):
        """Extract nodes from 3D mesh and verify shape."""
        manager = MFEMManager(str(beam_hex_mesh_file))

        nodes = manager.get_nodes()

        assert nodes is not None, "Nodes should not be None"
        assert nodes.shape[1] == 3, "beam-hex is 3D mesh"

    def test_dimension_property(self, beam_quad_mesh_file, beam_hex_mesh_file):
        """Test dimension property for 2D and 3D meshes."""
        manager_2d = MFEMManager(str(beam_quad_mesh_file))
        manager_3d = MFEMManager(str(beam_hex_mesh_file))

        assert manager_2d.dimension == 2, "beam-quad should be 2D"
        assert manager_3d.dimension == 3, "beam-hex should be 3D"

    def test_num_nodes_property(self, beam_quad_mesh_file):
        """Test num_nodes property."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        num_nodes = manager.num_nodes
        nodes = manager.get_nodes()

        assert num_nodes == len(nodes), "num_nodes should match node array length"
        assert num_nodes > 0, "Should have at least one node"

    def test_update_nodes_all(self, beam_quad_mesh_file):
        """Test updating all node coordinates."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        # Get original nodes
        original_nodes = manager.get_nodes().copy()

        # Scale all nodes by 1.5
        new_nodes = original_nodes * 1.5
        manager.update_nodes(new_nodes)

        # Verify update
        updated_nodes = manager.get_nodes()
        assert np.allclose(updated_nodes, new_nodes), "Nodes should be updated"
        assert manager.is_modified, "Mesh should be marked as modified"

    def test_update_nodes_specific(self, beam_quad_mesh_file):
        """Test updating specific nodes by ID."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        original_nodes = manager.get_nodes().copy()

        # Update first two nodes
        node_ids = np.array([0, 1])
        new_coords = np.array([[10.0, 10.0], [20.0, 20.0]])

        manager.update_nodes(new_coords, node_ids=node_ids)

        # Verify updates
        updated_nodes = manager.get_nodes()
        assert np.allclose(updated_nodes[0], [10.0, 10.0])
        assert np.allclose(updated_nodes[1], [20.0, 20.0])

        # Verify other nodes unchanged
        assert np.allclose(updated_nodes[2:], original_nodes[2:])


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestMFEMManagerElementExtraction:
    """Test cases for element extraction."""

    def test_extract_elements(self, beam_quad_mesh_file):
        """Test getting elements from mesh."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        elements = manager.get_elements()
        element_ids = manager.get_element_ids()

        assert elements is not None, "Elements should not be None"
        assert element_ids is not None, "Element IDs should not be None"

        assert isinstance(elements, np.ndarray), "Elements should be NumPy array"
        assert isinstance(element_ids, np.ndarray), "Element IDs should be NumPy array"

        assert len(element_ids) == len(elements), "Element IDs should match element count"

    def test_num_elements_property(self, beam_quad_mesh_file):
        """Test num_elements property."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        num_elements = manager.num_elements
        elements = manager.get_elements()

        assert num_elements == len(elements), "num_elements should match element array length"
        assert num_elements > 0, "Should have at least one element"

    def test_get_element_type(self, beam_quad_mesh_file):
        """Test getting element type."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        # beam-quad uses quadrilateral elements
        elem_type = manager.get_element_type(0)
        assert elem_type == "QUADRILATERAL", f"Expected QUADRILATERAL, got {elem_type}"

    def test_get_element_attributes(self, beam_quad_mesh_file):
        """Test getting element attributes (material IDs)."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        attrs = manager.get_element_attributes()
        assert isinstance(attrs, np.ndarray)
        assert len(attrs) == manager.num_elements


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestMFEMManagerSave:
    """Test cases for saving meshes."""

    def test_save_mesh(self, beam_quad_mesh_file, tmp_output_mesh):
        """Test saving mesh to file."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        # Modify nodes
        nodes = manager.get_nodes()
        manager.update_nodes(nodes * 1.1)

        # Save
        output_path = manager.save(tmp_output_mesh)

        assert Path(output_path).exists(), "Output file should be created"

        # Re-load and verify
        new_manager = MFEMManager(str(output_path))
        new_nodes = new_manager.get_nodes()

        assert np.allclose(new_nodes, nodes * 1.1, rtol=1e-6)

    def test_save_creates_parent_directory(self, beam_quad_mesh_file, tmp_path):
        """Test that save creates parent directories if needed."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        output_path = tmp_path / "nested" / "dir" / "output.mesh"
        manager.save(output_path)

        assert output_path.exists(), "Output file should be created"


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestMFEMManagerBoundingBox:
    """Test cases for bounding box and centroid."""

    def test_get_bounding_box(self, beam_quad_mesh_file):
        """Test getting mesh bounding box."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        min_coords, max_coords = manager.get_bounding_box()

        assert len(min_coords) == manager.dimension
        assert len(max_coords) == manager.dimension

        # Max should be >= min in all dimensions
        assert np.all(max_coords >= min_coords)

    def test_get_centroid(self, beam_quad_mesh_file):
        """Test getting mesh centroid."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        centroid = manager.get_centroid()

        assert len(centroid) == manager.dimension

        # Centroid should be within bounding box
        min_coords, max_coords = manager.get_bounding_box()
        assert np.all(centroid >= min_coords)
        assert np.all(centroid <= max_coords)


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestMFEMManagerBoundary:
    """Test cases for boundary information."""

    def test_get_boundary_attributes(self, beam_quad_mesh_file):
        """Test getting boundary attributes."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        bdr_attrs = manager.get_boundary_attributes()

        assert isinstance(bdr_attrs, np.ndarray)
        assert len(bdr_attrs) > 0, "Should have boundary elements"


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestMFEMManagerRefinement:
    """Test cases for mesh refinement."""

    def test_uniform_refinement(self, beam_quad_mesh_file):
        """Test uniform mesh refinement."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        original_num_elements = manager.num_elements
        original_num_nodes = manager.num_nodes

        # Refine once
        manager.refine_uniformly(times=1)

        # Should have more nodes and elements after refinement
        assert manager.num_nodes > original_num_nodes
        assert manager.num_elements > original_num_elements
        assert manager.is_modified


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestMFEMManagerErrorHandling:
    """Test cases for error handling."""

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            MFEMManager("nonexistent_file.mesh")

    def test_update_nodes_wrong_shape(self, beam_quad_mesh_file):
        """Test updating nodes with incorrect shape raises error."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        # Try to update with wrong number of nodes
        wrong_coords = np.array([[0.0, 0.0]])  # Only 1 node

        with pytest.raises(ValueError):
            manager.update_nodes(wrong_coords)

    def test_update_nodes_invalid_id(self, beam_quad_mesh_file):
        """Test updating with invalid node ID raises error."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        # Try to update with invalid node ID
        invalid_ids = np.array([9999])
        coords = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError):
            manager.update_nodes(coords, node_ids=invalid_ids)

    def test_get_element_type_invalid_id(self, beam_quad_mesh_file):
        """Test getting element type with invalid ID raises error."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        with pytest.raises(ValueError):
            manager.get_element_type(9999)


@pytest.mark.skipif(not MFEM_AVAILABLE, reason="PyMFEM not installed")
class TestMeshManagerInterface:
    """Test that MFEMManager properly implements MeshManager interface."""

    def test_implements_mesh_manager(self, beam_quad_mesh_file):
        """Test that MFEMManager is a MeshManager instance."""
        manager = MFEMManager(str(beam_quad_mesh_file))
        assert isinstance(manager, MeshManager)

    def test_all_abstract_methods_implemented(self, beam_quad_mesh_file):
        """Test that all abstract methods are implemented."""
        manager = MFEMManager(str(beam_quad_mesh_file))

        # These should not raise NotImplementedError
        manager.get_nodes()
        manager.get_node_ids()
        manager.get_elements()
        manager.get_element_ids()
        _ = manager.dimension
        _ = manager.num_nodes
        _ = manager.num_elements
