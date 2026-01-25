"""
Pre-flight Integrity Checker for Abaqus Models.

This module serves as a support tool for the Evaluator and Critic agents.
It checks the integrity of modified models before execution, including:
1) ID Consistency (e.g., BCs refer to existing Nsets)
2) Geometric Validity (overlapping nodes, aspect ratios)
3) Unit Consistency
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass

from manager import AbaqusManager
from schema import Nodes, Elements, BoundaryCondition, NodeSet, ElementSet


@dataclass
class ValidationReport:
    """
    Report containing validation results.
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.details = {}
    
    def add_error(self, error: str):
        """Add an error to the report."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning to the report."""
        self.warnings.append(warning)
    
    def get_summary(self) -> str:
        """Get a summary of the validation report."""
        lines = []
        lines.append(f"Validation Status: {'PASS' if self.is_valid else 'FAIL'}")
        lines.append(f"Errors: {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            lines.append("\nErrors:")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")
        
        return "\n".join(lines)


class AbaqusValidator:
    """
    Validator class that performs integrity checks on Abaqus models.
    """
    
    # Geometric validation thresholds
    MIN_NODE_DISTANCE = 1e-10  # Minimum distance between nodes
    MAX_ASPECT_RATIO = 100.0   # Maximum element aspect ratio
    MIN_JACOBIAN = 1e-6        # Minimum element Jacobian
    
    def __init__(self, manager: AbaqusManager):
        """
        Initialize the validator with a manager instance.
        
        Args:
            manager: AbaqusManager instance to validate
        """
        self.manager = manager
        self.report = ValidationReport()
    
    def validate_all(self) -> ValidationReport:
        """
        Perform all validation checks.
        
        Returns:
            ValidationReport with all results
        """
        self.report = ValidationReport()
        
        # Run all validation checks
        self._check_id_consistency()
        self._check_geometric_validity()
        self._check_unit_consistency()
        self._check_data_integrity()
        
        return self.report
    
    def _check_id_consistency(self):
        """Check ID consistency (BCs refer to existing sets, etc.)."""
        # Check boundary conditions reference existing node sets
        for bc_name, bc in self.manager.boundary_conditions.items():
            set_name = bc.data.get('set_name', '')
            if set_name:
                if set_name not in self.manager.node_sets:
                    self.report.add_error(
                        f"Boundary condition '{bc_name}' references node set '{set_name}' "
                        f"which does not exist"
                    )
                else:
                    # Verify nodes in the set exist
                    nset = self.manager.node_sets[set_name]
                    if 'node_ids' in nset.data:
                        node_ids = nset.data['node_ids']
                        if self.manager.nodes:
                            existing_ids = set(self.manager.nodes.ids)
                            missing_ids = [nid for nid in node_ids if nid not in existing_ids]
                            if missing_ids:
                                self.report.add_error(
                                    f"Node set '{set_name}' contains node IDs that don't exist: "
                                    f"{missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}"
                                )
        
        # Check element sets reference existing elements
        for elset_name, elset in self.manager.element_sets.items():
            if 'element_ids' in elset.data:
                element_ids = elset.data['element_ids']
                if self.manager.elements:
                    existing_ids = set(self.manager.elements.ids)
                    missing_ids = [eid for eid in element_ids if eid not in existing_ids]
                    if missing_ids:
                        self.report.add_error(
                            f"Element set '{elset_name}' contains element IDs that don't exist: "
                            f"{missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}"
                        )
        
        # Check element connectivity references existing nodes
        if self.manager.elements and self.manager.nodes:
            existing_node_ids = set(self.manager.nodes.ids)
            invalid_elements = []
            
            for i in range(len(self.manager.elements.ids)):
                element_id = self.manager.elements.ids[i]
                connectivity = self.manager.elements.data[i]
                
                missing_nodes = [nid for nid in connectivity if nid not in existing_node_ids]
                if missing_nodes:
                    invalid_elements.append((element_id, missing_nodes))
            
            if invalid_elements:
                error_msg = (
                    f"Found {len(invalid_elements)} element(s) with invalid node references. "
                    f"Examples: {invalid_elements[:5]}"
                )
                self.report.add_error(error_msg)
    
    def _check_geometric_validity(self):
        """Check geometric validity (overlapping nodes, aspect ratios, etc.)."""
        if self.manager.nodes is None:
            return
        
        nodes = self.manager.nodes
        coords = nodes.data
        
        # Check for overlapping nodes
        if len(coords) > 1:
            from scipy.spatial.distance import cdist
            try:
                # Compute pairwise distances
                distances = cdist(coords, coords)
                # Set diagonal to large value to ignore self-distances
                np.fill_diagonal(distances, np.inf)
                
                # Find nodes that are too close
                too_close = np.where(distances < self.MIN_NODE_DISTANCE)
                if len(too_close[0]) > 0:
                    pairs = list(zip(too_close[0][:10], too_close[1][:10]))
                    node_pairs = [(nodes.ids[i], nodes.ids[j]) for i, j in pairs]
                    self.report.add_warning(
                        f"Found {len(too_close[0])} pair(s) of nodes that are very close "
                        f"(< {self.MIN_NODE_DISTANCE}). Examples: {node_pairs[:5]}"
                    )
            except ImportError:
                # scipy not available, use simpler check
                self.report.add_warning(
                    "scipy not available, skipping detailed node distance check"
                )
        
        # Check element aspect ratios and Jacobians
        if self.manager.elements is not None:
            elements = self.manager.elements
            invalid_elements = []
            high_aspect_ratio_elements = []
            
            for i in range(len(elements.ids)):
                element_id = elements.ids[i]
                connectivity = elements.data[i]
                
                # Get node coordinates for this element
                try:
                    node_coords = []
                    for node_id in connectivity:
                        idx = np.where(nodes.ids == node_id)[0]
                        if len(idx) > 0:
                            node_coords.append(coords[idx[0]])
                    
                    if len(node_coords) >= 2:
                        node_coords = np.array(node_coords)
                        
                        # Calculate aspect ratio (simplified for 2D/3D)
                        if len(node_coords[0]) == 2:  # 2D
                            # For 2D elements, calculate bounding box
                            x_range = np.max(node_coords[:, 0]) - np.min(node_coords[:, 0])
                            y_range = np.max(node_coords[:, 1]) - np.min(node_coords[:, 1])
                            
                            if min(x_range, y_range) > 0:
                                aspect_ratio = max(x_range, y_range) / min(x_range, y_range)
                                if aspect_ratio > self.MAX_ASPECT_RATIO:
                                    high_aspect_ratio_elements.append((element_id, aspect_ratio))
                        elif len(node_coords[0]) == 3:  # 3D
                            # For 3D elements, calculate bounding box
                            x_range = np.max(node_coords[:, 0]) - np.min(node_coords[:, 0])
                            y_range = np.max(node_coords[:, 1]) - np.min(node_coords[:, 1])
                            z_range = np.max(node_coords[:, 2]) - np.min(node_coords[:, 2])
                            
                            ranges = [r for r in [x_range, y_range, z_range] if r > 0]
                            if len(ranges) >= 2:
                                aspect_ratio = max(ranges) / min(ranges)
                                if aspect_ratio > self.MAX_ASPECT_RATIO:
                                    high_aspect_ratio_elements.append((element_id, aspect_ratio))
                
                except Exception as e:
                    invalid_elements.append((element_id, str(e)))
            
            if invalid_elements:
                self.report.add_error(
                    f"Found {len(invalid_elements)} element(s) with invalid geometry. "
                    f"Examples: {invalid_elements[:5]}"
                )
            
            if high_aspect_ratio_elements:
                self.report.add_warning(
                    f"Found {len(high_aspect_ratio_elements)} element(s) with high aspect ratio "
                    f"(> {self.MAX_ASPECT_RATIO}). Examples: {high_aspect_ratio_elements[:5]}"
                )
    
    def _check_unit_consistency(self):
        """Check unit consistency (basic checks)."""
        # Check material properties for reasonable values
        for material_name, material in self.manager.materials.items():
            if 'elastic' in material.data:
                elastic = material.data['elastic']
                E = elastic.get('E', 0.0)
                nu = elastic.get('nu', 0.0)
                
                # Check elastic modulus is positive
                if E <= 0:
                    self.report.add_error(
                        f"Material '{material_name}' has non-positive elastic modulus: {E}"
                    )
                
                # Check Poisson's ratio is in valid range
                if nu < 0 or nu >= 0.5:
                    self.report.add_error(
                        f"Material '{material_name}' has invalid Poisson's ratio: {nu} "
                        f"(must be 0 <= nu < 0.5)"
                    )
                
                # Warn about unusual values
                if E > 1e12:
                    self.report.add_warning(
                        f"Material '{material_name}' has very large elastic modulus: {E}. "
                        f"Check units (should be in Pa or consistent units)."
                    )
        
        # Check node coordinates for reasonable scale
        if self.manager.nodes:
            coords = self.manager.nodes.data
            if len(coords) > 0:
                coord_ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
                max_range = np.max(coord_ranges)
                
                if max_range > 1e6:
                    self.report.add_warning(
                        f"Model has very large dimensions (max range: {max_range}). "
                        f"Check units."
                    )
                elif max_range < 1e-6:
                    self.report.add_warning(
                        f"Model has very small dimensions (max range: {max_range}). "
                        f"Check units."
                    )
    
    def _check_data_integrity(self):
        """Check basic data integrity."""
        # Check nodes
        if self.manager.nodes:
            errors = self.manager.nodes.validate()
            for error in errors:
                self.report.add_error(f"Node data: {error}")
        
        # Check elements
        if self.manager.elements:
            errors = self.manager.elements.validate()
            for error in errors:
                self.report.add_error(f"Element data: {error}")
        
        # Check materials
        for material_name, material in self.manager.materials.items():
            errors = material.validate()
            for error in errors:
                self.report.add_error(f"Material '{material_name}': {error}")
        
        # Check boundary conditions
        for bc_name, bc in self.manager.boundary_conditions.items():
            errors = bc.validate()
            for error in errors:
                self.report.add_error(f"Boundary condition '{bc_name}': {error}")
        
        # Check node sets
        for nset_name, nset in self.manager.node_sets.items():
            errors = nset.validate()
            for error in errors:
                self.report.add_error(f"Node set '{nset_name}': {error}")
        
        # Check element sets
        for elset_name, elset in self.manager.element_sets.items():
            errors = elset.validate()
            for error in errors:
                self.report.add_error(f"Element set '{elset_name}': {error}")


def validate_model(manager: AbaqusManager) -> ValidationReport:
    """
    Convenience function to validate an Abaqus model.
    
    Args:
        manager: AbaqusManager instance to validate
        
    Returns:
        ValidationReport with validation results
    """
    validator = AbaqusValidator(manager)
    return validator.validate_all()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        manager = AbaqusManager(file_path)
        
        validator = AbaqusValidator(manager)
        report = validator.validate_all()
        
        print(report.get_summary())
        
        if not report.is_valid:
            sys.exit(1)
    else:
        print("Usage: python validator.py <path_to_inp_file>")
