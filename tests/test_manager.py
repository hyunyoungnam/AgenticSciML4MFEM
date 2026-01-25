"""
Tests for manager.py module.

This module tests the interaction between AI agents and the data through
the manager API, including:
1. Extracting nodes into NumPy arrays
2. Updating material properties
3. Error handling for non-existent entities
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from manager import AbaqusManager


class TestManagerNodeExtraction:
    """Test cases for node extraction and manipulation."""
    
    def test_extract_nodes_into_numpy_array(self, tmp_inp_file):
        """Extract nodes into NumPy array and verify shape matches input."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        # Get nodes
        nodes = manager.get_nodes()
        node_ids = manager.get_node_ids()
        
        assert nodes is not None, "Nodes should not be None"
        assert node_ids is not None, "Node IDs should not be None"
        
        # Verify shape
        assert isinstance(nodes, np.ndarray), "Nodes should be NumPy array"
        assert isinstance(node_ids, np.ndarray), "Node IDs should be NumPy array"
        
        # Base inp has 6 nodes with 2D coordinates
        assert nodes.shape == (6, 2), f"Expected shape (6, 2), got {nodes.shape}"
        assert node_ids.shape == (6,), f"Expected shape (6,), got {node_ids.shape}"
        
        # Verify data types
        assert nodes.dtype in [np.float64, np.float32], "Coordinates should be float"
        assert node_ids.dtype in [np.int32, np.int64], "Node IDs should be integer"
    
    def test_get_node_coordinates(self, tmp_inp_file):
        """Test getting coordinates for a specific node."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        # Get coordinates for node 1
        coords = manager.get_node_coordinates(1)
        
        assert coords is not None, "Coordinates should not be None"
        assert isinstance(coords, np.ndarray), "Coordinates should be NumPy array"
        assert coords.shape == (2,), f"Expected shape (2,), got {coords.shape}"
        assert np.allclose(coords, [0.0, 0.0]), f"Expected [0.0, 0.0], got {coords}"
    
    def test_get_nonexistent_node_coordinates(self, tmp_inp_file):
        """Test getting coordinates for non-existent node returns None."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        coords = manager.get_node_coordinates(999)
        assert coords is None, "Non-existent node should return None"
    
    def test_update_nodes(self, tmp_inp_file):
        """Test updating node coordinates."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        # Get original nodes
        original_nodes = manager.get_nodes().copy()
        
        # Scale all nodes by 1.1
        new_nodes = original_nodes * 1.1
        manager.update_nodes(new_nodes)
        
        # Verify update
        updated_nodes = manager.get_nodes()
        assert np.allclose(updated_nodes, new_nodes), "Nodes should be updated"
        assert manager.nodes.modified, "Nodes should be marked as modified"
        assert 'NODE' in manager.get_modified_sections(), "NODE should be in modified sections"


class TestManagerMaterialOperations:
    """Test cases for material property operations."""
    
    def test_get_material(self, tmp_inp_file):
        """Test getting material by name."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        material = manager.get_material('Material-1')
        
        assert material is not None, "Material should not be None"
        assert material.name == 'Material-1', "Material name should match"
        assert 'elastic' in material.data, "Material should have elastic properties"
    
    def test_update_material_property(self, tmp_inp_file):
        """Update material property and verify change is reflected."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        # Get original material
        material = manager.get_material('Material-1')
        original_E = material.data['elastic']['E']
        
        # Update Young's Modulus
        new_E = 250000.0
        manager.update_material('Material-1', {'elastic': {'E': new_E, 'nu': 0.3}})
        
        # Verify update
        updated_material = manager.get_material('Material-1')
        assert updated_material.data['elastic']['E'] == new_E, \
            f"Expected E={new_E}, got {updated_material.data['elastic']['E']}"
        assert updated_material.modified, "Material should be marked as modified"
        assert 'MATERIAL' in manager.get_modified_sections(), \
            "MATERIAL should be in modified sections"
    
    def test_update_nonexistent_material(self, tmp_inp_file):
        """Attempt to update non-existent material should raise exception."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        with pytest.raises(ValueError, match="Material 'NonExistent' not found"):
            manager.update_material('NonExistent', {'elastic': {'E': 200000.0}})
    
    def test_get_nonexistent_material(self, tmp_inp_file):
        """Getting non-existent material should return None."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        material = manager.get_material('NonExistent')
        assert material is None, "Non-existent material should return None"


class TestManagerBoundaryConditionOperations:
    """Test cases for boundary condition operations."""
    
    def test_get_boundary_condition(self, tmp_inp_file):
        """Test getting boundary condition."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        # Should have at least one boundary condition
        bc_names = list(manager.boundary_conditions.keys())
        assert len(bc_names) > 0, "Should have at least one boundary condition"
        
        bc = manager.get_boundary_condition(bc_names[0])
        assert bc is not None, "Boundary condition should not be None"
        assert 'set_name' in bc.data, "BC should have set_name"
        assert 'type' in bc.data, "BC should have type"
    
    def test_modify_boundary_condition(self, tmp_inp_file):
        """Test modifying boundary condition."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        bc_names = list(manager.boundary_conditions.keys())
        if len(bc_names) > 0:
            bc_name = bc_names[0]
            original_type = manager.boundary_conditions[bc_name].data['type']
            
            # Modify BC type
            new_type = 'YSYMM'
            manager.modify_boundary_condition(bc_name, new_type, param='type')
            
            # Verify update
            updated_bc = manager.get_boundary_condition(bc_name)
            assert updated_bc.data['type'] == new_type, \
                f"Expected type={new_type}, got {updated_bc.data['type']}"
            assert updated_bc.modified, "BC should be marked as modified"
            assert 'BOUNDARY' in manager.get_modified_sections(), \
                "BOUNDARY should be in modified sections"
    
    def test_modify_nonexistent_boundary_condition(self, tmp_inp_file):
        """Attempt to modify non-existent BC should raise exception."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        with pytest.raises(ValueError, match="Boundary condition 'NonExistent' not found"):
            manager.modify_boundary_condition('NonExistent', 'XSYMM', param='type')


class TestManagerNodeSetOperations:
    """Test cases for node set operations."""
    
    def test_get_node_set(self, tmp_inp_file):
        """Test getting node set."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        nset = manager.get_node_set('Set-1')
        assert nset is not None, "Node set should not be None"
        assert nset.name == 'Set-1', "Node set name should match"
    
    def test_get_nonexistent_node_set(self, tmp_inp_file):
        """Getting non-existent node set should return None."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        nset = manager.get_node_set('NonExistent')
        assert nset is None, "Non-existent node set should return None"


class TestManagerElementSetOperations:
    """Test cases for element set operations."""
    
    def test_get_element_set(self, tmp_inp_file):
        """Test getting element set."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        elset = manager.get_element_set('Set-1')
        assert elset is not None, "Element set should not be None"
        assert elset.name == 'Set-1', "Element set name should match"
    
    def test_get_nonexistent_element_set(self, tmp_inp_file):
        """Getting non-existent element set should return None."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        elset = manager.get_element_set('NonExistent')
        assert elset is None, "Non-existent element set should return None"


class TestManagerElementOperations:
    """Test cases for element operations."""
    
    def test_get_elements(self, tmp_inp_file):
        """Test getting elements."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        elements = manager.get_elements()
        element_ids = manager.get_element_ids()
        
        assert elements is not None, "Elements should not be None"
        assert element_ids is not None, "Element IDs should not be None"
        
        assert isinstance(elements, np.ndarray), "Elements should be NumPy array"
        assert isinstance(element_ids, np.ndarray), "Element IDs should be NumPy array"
        
        # Base inp has 2 elements
        assert len(element_ids) == 2, f"Expected 2 elements, got {len(element_ids)}"


class TestManagerErrorHandling:
    """Test cases for error handling and edge cases."""
    
    def test_update_nodes_with_wrong_shape(self, tmp_inp_file):
        """Test updating nodes with incorrect shape raises error."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        # Try to update with wrong number of coordinates
        wrong_coords = np.array([[0.0, 0.0]])  # Only 1 node instead of 6
        
        with pytest.raises(ValueError, match="Number of coordinates"):
            manager.update_nodes(wrong_coords)
    
    def test_update_nodes_with_node_ids(self, tmp_inp_file):
        """Test updating specific nodes by ID."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        # Update specific nodes
        node_ids = np.array([1, 2])
        new_coords = np.array([[10.0, 10.0], [20.0, 20.0]])
        
        manager.update_nodes(new_coords, node_ids=node_ids)
        
        # Verify updates
        assert np.allclose(manager.get_node_coordinates(1), [10.0, 10.0])
        assert np.allclose(manager.get_node_coordinates(2), [20.0, 20.0])
    
    def test_get_modified_sections(self, tmp_inp_file):
        """Test getting list of modified sections."""
        manager = AbaqusManager(str(tmp_inp_file))
        
        # Initially no modifications
        modified = manager.get_modified_sections()
        assert len(modified) == 0, "Initially no sections should be modified"
        
        # Make a modification
        nodes = manager.get_nodes()
        manager.update_nodes(nodes * 1.1)
        
        modified = manager.get_modified_sections()
        assert 'NODE' in modified, "NODE should be in modified sections after update"
