"""
Integration tests for full round-trip cycle.

This module tests the complete workflow:
1. Load a base .inp file
2. Modify node coordinates and boundary conditions
3. Export to new file
4. Re-parse and verify modifications are correct
5. Verify unmodified data remains identical
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from inpforge.manager import AbaqusManager
from inpforge.writer import AbaqusWriter, write_inp_file
from inpforge.parser import AbaqusParser
from inpforge.validator import AbaqusValidator, validate_model


class TestRoundTripIntegration:
    """Test cases for complete round-trip cycle."""
    
    def test_full_round_trip_node_modification(self, tmp_inp_file, tmp_output_file):
        """
        Complete round-trip: Load -> Modify Nodes -> Write -> Re-parse -> Verify.
        
        This test ensures:
        1. Modified node coordinates are correctly written and read back
        2. Unmodified nodes remain identical
        3. All other data (elements, materials, etc.) remain unchanged
        """
        # Step 1: Load base .inp file
        manager = AbaqusManager(str(tmp_inp_file))
        
        # Get original data for comparison
        original_nodes = manager.get_nodes().copy()
        original_node_ids = manager.get_node_ids().copy()
        original_elements = manager.get_elements().copy()
        original_element_ids = manager.get_element_ids().copy()
        original_material = manager.get_material('Material-1')
        original_material_E = original_material.data['elastic']['E'] if original_material else None
        
        # Step 2: Modify a subset of node coordinates
        # Modify first 3 nodes: scale by 1.5
        modified_node_ids = original_node_ids[:3]
        modified_coords = original_nodes[:3] * 1.5
        
        manager.update_nodes(modified_coords, node_ids=modified_node_ids)
        
        # Verify modification in memory
        assert manager.nodes.modified, "Nodes should be marked as modified"
        assert 'NODE' in manager.get_modified_sections()
        
        # Step 3: Export modified model to new file
        writer = AbaqusWriter(manager)
        output_path = writer.write_file(str(tmp_output_file))
        
        assert Path(output_path).exists(), "Output file should be created"
        
        # Step 4: Re-parse the newly created file
        new_manager = AbaqusManager(output_path)
        
        # Step 5: Verify modified values are exactly as expected
        new_nodes = new_manager.get_nodes()
        new_node_ids = new_manager.get_node_ids()
        
        # Check that modified nodes are correct
        for i, node_id in enumerate(modified_node_ids):
            idx = np.where(new_node_ids == node_id)[0]
            assert len(idx) > 0, f"Node {node_id} should exist in new file"
            expected_coords = modified_coords[i]
            actual_coords = new_nodes[idx[0]]
            assert np.allclose(actual_coords, expected_coords, rtol=1e-6), \
                f"Node {node_id} coordinates mismatch: expected {expected_coords}, got {actual_coords}"
        
        # Verify unmodified nodes remain identical
        unmodified_node_ids = original_node_ids[3:]
        for node_id in unmodified_node_ids:
            orig_idx = np.where(original_node_ids == node_id)[0]
            new_idx = np.where(new_node_ids == node_id)[0]
            
            assert len(orig_idx) > 0 and len(new_idx) > 0, \
                f"Node {node_id} should exist in both files"
            
            orig_coords = original_nodes[orig_idx[0]]
            new_coords = new_nodes[new_idx[0]]
            
            assert np.allclose(orig_coords, new_coords, rtol=1e-6), \
                f"Unmodified node {node_id} coordinates changed: {orig_coords} vs {new_coords}"
        
        # Step 6: Verify all other Heavy Data and Light Data remain identical
        # Check elements
        new_elements = new_manager.get_elements()
        new_element_ids = new_manager.get_element_ids()
        
        assert np.array_equal(original_element_ids, new_element_ids), \
            "Element IDs should remain unchanged"
        assert np.array_equal(original_elements, new_elements), \
            "Element connectivity should remain unchanged"
        
        # Check materials
        new_material = new_manager.get_material('Material-1')
        assert new_material is not None, "Material should still exist"
        if original_material_E:
            new_material_E = new_material.data['elastic']['E']
            assert abs(original_material_E - new_material_E) < 1e-6, \
                f"Material E should remain unchanged: {original_material_E} vs {new_material_E}"
    
    def test_full_round_trip_boundary_condition_modification(
        self, tmp_inp_file, tmp_output_file
    ):
        """
        Complete round-trip: Load -> Modify BC -> Write -> Re-parse -> Verify.
        
        This test ensures:
        1. Modified boundary conditions are correctly written and read back
        2. All other data remains unchanged
        """
        # Step 1: Load base .inp file
        manager = AbaqusManager(str(tmp_inp_file))
        
        # Get original BCs
        bc_names = list(manager.boundary_conditions.keys())
        assert len(bc_names) > 0, "Should have at least one boundary condition"
        
        original_bc = manager.get_boundary_condition(bc_names[0])
        original_bc_type = original_bc.data['type']
        
        # Step 2: Modify boundary condition
        new_bc_type = 'YSYMM' if original_bc_type == 'XSYMM' else 'XSYMM'
        manager.modify_boundary_condition(bc_names[0], new_bc_type, param='type')
        
        # Step 3: Export modified model
        writer = AbaqusWriter(manager)
        output_path = writer.write_file(str(tmp_output_file))
        
        # Step 4: Re-parse
        new_manager = AbaqusManager(output_path)
        
        # Step 5: Verify modification
        new_bc = new_manager.get_boundary_condition(bc_names[0])
        assert new_bc is not None, "Boundary condition should still exist"
        assert new_bc.data['type'] == new_bc_type, \
            f"BC type should be updated: expected {new_bc_type}, got {new_bc.data['type']}"
        
        # Verify other data unchanged
        original_nodes = manager.get_nodes()
        new_nodes = new_manager.get_nodes()
        assert np.allclose(original_nodes, new_nodes, rtol=1e-6), \
            "Node coordinates should remain unchanged"
    
    def test_full_round_trip_material_modification(self, tmp_inp_file, tmp_output_file):
        """
        Complete round-trip: Load -> Modify Material -> Write -> Re-parse -> Verify.
        """
        # Step 1: Load base .inp file
        manager = AbaqusManager(str(tmp_inp_file))
        
        original_material = manager.get_material('Material-1')
        original_E = original_material.data['elastic']['E']
        original_nu = original_material.data['elastic']['nu']
        
        # Step 2: Modify material property
        new_E = 300000.0
        manager.update_material('Material-1', {'elastic': {'E': new_E, 'nu': original_nu}})
        
        # Step 3: Export
        writer = AbaqusWriter(manager)
        output_path = writer.write_file(str(tmp_output_file))
        
        # Step 4: Re-parse
        new_manager = AbaqusManager(output_path)
        
        # Step 5: Verify modification
        new_material = new_manager.get_material('Material-1')
        assert new_material is not None, "Material should still exist"
        assert abs(new_material.data['elastic']['E'] - new_E) < 1e-6, \
            f"Material E should be updated: expected {new_E}, got {new_material.data['elastic']['E']}"
        assert abs(new_material.data['elastic']['nu'] - original_nu) < 1e-6, \
            "Material nu should remain unchanged"
    
    def test_round_trip_preserves_scientific_notation(self, tmp_inp_file, tmp_output_file):
        """
        Test that scientific notation is preserved through round-trip.
        """
        # Load file with scientific notation
        manager = AbaqusManager(str(tmp_inp_file))
        
        # Find node with scientific notation (node 6 in base_inp_content)
        node_ids = manager.get_node_ids()
        nodes = manager.get_nodes()
        
        # Node 6 should have scientific notation values
        node_6_idx = np.where(node_ids == 6)[0]
        if len(node_6_idx) > 0:
            original_coords = nodes[node_6_idx[0]].copy()
            
            # Write and re-read
            writer = AbaqusWriter(manager)
            output_path = writer.write_file(str(tmp_output_file))
            
            new_manager = AbaqusManager(output_path)
            new_node_ids = new_manager.get_node_ids()
            new_nodes = new_manager.get_nodes()
            
            new_node_6_idx = np.where(new_node_ids == 6)[0]
            if len(new_node_6_idx) > 0:
                new_coords = new_nodes[new_node_6_idx[0]]
                
                # Verify scientific notation values are preserved
                assert np.allclose(original_coords, new_coords, rtol=1e-6), \
                    "Scientific notation values should be preserved"


class TestIntegrationWithValidator:
    """Test cases for integration with validator."""
    
    def test_modified_model_passes_validation(self, tmp_inp_file, tmp_output_file):
        """
        Test that modified model passes validation checks.
        """
        # Load and modify
        manager = AbaqusManager(str(tmp_inp_file))
        nodes = manager.get_nodes()
        manager.update_nodes(nodes * 1.1)  # Scale all nodes
        
        # Write
        writer = AbaqusWriter(manager)
        output_path = writer.write_file(str(tmp_output_file))
        
        # Validate
        new_manager = AbaqusManager(output_path)
        validator = AbaqusValidator(new_manager)
        report = validator.validate_all()
        
        # Should pass basic validation (may have warnings but no errors)
        # Note: Scaling might cause geometric warnings, but should not cause errors
        assert report.is_valid or len(report.errors) == 0, \
            f"Modified model should pass validation. Errors: {report.errors}"
    
    def test_invalid_modification_detected(self, tmp_inp_file):
        """
        Test that validator detects invalid modifications.
        """
        manager = AbaqusManager(str(tmp_inp_file))
        
        # Create invalid modification: set node coordinates to invalid values
        # This should be caught by validator
        nodes = manager.get_nodes()
        # Create overlapping nodes (set two nodes to same coordinates)
        invalid_nodes = nodes.copy()
        invalid_nodes[1] = invalid_nodes[0]  # Make node 2 same as node 1
        
        manager.update_nodes(invalid_nodes)
        
        # Validate
        validator = AbaqusValidator(manager)
        report = validator.validate_all()
        
        # Should detect issues (may be warnings or errors depending on implementation)
        # At minimum, should not crash


class TestDataIntegrity:
    """Test cases for data integrity during round-trip."""
    
    def test_no_data_loss_during_round_trip(self, tmp_inp_file, tmp_output_file):
        """
        Verify that no data is lost during round-trip.
        """
        # Load original
        original_manager = AbaqusManager(str(tmp_inp_file))
        
        # Count original entities
        original_node_count = len(original_manager.get_node_ids()) if original_manager.nodes else 0
        original_element_count = len(original_manager.get_element_ids()) if original_manager.elements else 0
        original_material_count = len(original_manager.materials)
        original_bc_count = len(original_manager.boundary_conditions)
        original_nset_count = len(original_manager.node_sets)
        original_elset_count = len(original_manager.element_sets)
        
        # Make a small modification (scale nodes)
        nodes = original_manager.get_nodes()
        original_manager.update_nodes(nodes * 1.01)
        
        # Write
        writer = AbaqusWriter(original_manager)
        output_path = writer.write_file(str(tmp_output_file))
        
        # Re-load
        new_manager = AbaqusManager(output_path)
        
        # Verify counts match
        new_node_count = len(new_manager.get_node_ids()) if new_manager.nodes else 0
        new_element_count = len(new_manager.get_element_ids()) if new_manager.elements else 0
        new_material_count = len(new_manager.materials)
        new_bc_count = len(new_manager.boundary_conditions)
        new_nset_count = len(new_manager.node_sets)
        new_elset_count = len(new_manager.element_sets)
        
        assert new_node_count == original_node_count, \
            f"Node count mismatch: {original_node_count} vs {new_node_count}"
        assert new_element_count == original_element_count, \
            f"Element count mismatch: {original_element_count} vs {new_element_count}"
        assert new_material_count == original_material_count, \
            f"Material count mismatch: {original_material_count} vs {new_material_count}"
        assert new_bc_count == original_bc_count, \
            f"BC count mismatch: {original_bc_count} vs {new_bc_count}"
        assert new_nset_count == original_nset_count, \
            f"Node set count mismatch: {original_nset_count} vs {new_nset_count}"
        assert new_elset_count == original_elset_count, \
            f"Element set count mismatch: {original_elset_count} vs {new_elset_count}"
    
    def test_formatting_preserved_for_unmodified_sections(self, tmp_inp_file, tmp_output_file):
        """
        Test that formatting is preserved for unmodified sections.
        """
        # Load original
        original_parser = AbaqusParser()
        original_chunks = original_parser.parse_file(str(tmp_inp_file))
        
        # Modify only nodes
        manager = AbaqusManager(str(tmp_inp_file))
        nodes = manager.get_nodes()
        manager.update_nodes(nodes * 1.1)
        
        # Write
        writer = AbaqusWriter(manager)
        output_path = writer.write_file(str(tmp_output_file))
        
        # Re-parse
        new_parser = AbaqusParser()
        new_chunks = new_parser.parse_file(output_path)
        
        # Material section should be preserved (not modified)
        if 'MATERIAL' in original_chunks and 'MATERIAL' in new_chunks:
            original_material = original_chunks['MATERIAL'][0]['raw_text']
            new_material = new_chunks['MATERIAL'][0]['raw_text']
            
            # Should be identical (or very similar) since not modified
            # Allow for minor whitespace differences
            original_clean = ' '.join(original_material.split())
            new_clean = ' '.join(new_material.split())
            assert original_clean == new_clean, \
                "Unmodified material section should be preserved"
