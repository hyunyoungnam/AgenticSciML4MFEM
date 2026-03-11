"""
Data Structure Definitions for Mesh Models.

This module defines data models using dataclasses:
- HeavyData: For large numerical arrays (Nodes, Elements) using NumPy
- LightData: For configuration-based data (Materials, BCs, Steps) using dictionaries
- PhysicsType: Enum for supported physics types
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union
import numpy as np
import re


class PhysicsType(Enum):
    """Enumeration of supported physics types for FEM analysis."""
    LINEAR_ELASTICITY = auto()
    HEAT_TRANSFER = auto()


@dataclass
class HeavyData:
    """
    Base class for large numerical data arrays (Nodes, Elements).
    Uses NumPy for memory efficiency.
    """
    ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    modified: bool = False
    
    def validate(self) -> List[str]:
        """
        Validate the data structure.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check IDs are integers
        if len(self.ids) > 0:
            if not np.all(np.equal(np.mod(self.ids, 1), 0)):
                errors.append("Node/Element IDs must be integers")
            
            if np.any(self.ids <= 0):
                errors.append("Node/Element IDs must be positive integers")
            
            if len(np.unique(self.ids)) != len(self.ids):
                errors.append("Node/Element IDs must be unique")
        
        # Check data dimensions
        if len(self.ids) > 0 and len(self.data) > 0:
            if len(self.ids) != len(self.data):
                errors.append("IDs and data arrays must have the same length")
        
        return errors


@dataclass
class Nodes(HeavyData):
    """
    Node data structure.
    - ids: Node IDs (1D array)
    - data: Node coordinates (N x 3 array for 3D, N x 2 for 2D)
    """
    
    def __post_init__(self):
        """Validate node data after initialization."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Node validation errors: {', '.join(errors)}")
    
    def validate(self) -> List[str]:
        """Validate node-specific data."""
        errors = super().validate()
        
        if len(self.data) > 0:
            # Check coordinates are floating point
            if self.data.dtype != np.float64 and self.data.dtype != np.float32:
                errors.append("Node coordinates must be floating point numbers")
            
            # Check coordinate dimensions (2D or 3D)
            if len(self.data.shape) == 2:
                if self.data.shape[1] not in [2, 3]:
                    errors.append("Node coordinates must be 2D (x, y) or 3D (x, y, z)")
            else:
                errors.append("Node coordinates must be 2D array (N x 2 or N x 3)")
        
        return errors
    
    def get_coordinates(self, node_id: int) -> np.ndarray:
        """Get coordinates for a specific node ID."""
        idx = np.where(self.ids == node_id)[0]
        if len(idx) == 0:
            raise ValueError(f"Node ID {node_id} not found")
        return self.data[idx[0]]
    
    def update_coordinates(self, node_id: int, coords: np.ndarray):
        """Update coordinates for a specific node ID."""
        idx = np.where(self.ids == node_id)[0]
        if len(idx) == 0:
            raise ValueError(f"Node ID {node_id} not found")
        self.data[idx[0]] = coords
        self.modified = True


@dataclass
class Elements(HeavyData):
    """
    Element data structure.
    - ids: Element IDs (1D array)
    - data: Element connectivity (N x M array, where M depends on element type)
    """
    element_type: str = ""
    
    def __post_init__(self):
        """Validate element data after initialization."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Element validation errors: {', '.join(errors)}")
    
    def validate(self) -> List[str]:
        """Validate element-specific data."""
        errors = super().validate()
        
        if len(self.data) > 0:
            # Check connectivity are integers
            if self.data.dtype != np.int32 and self.data.dtype != np.int64:
                errors.append("Element connectivity must be integers")
            
            # Check connectivity values are positive
            if np.any(self.data <= 0):
                errors.append("Element connectivity node IDs must be positive integers")
        
        return errors
    
    def get_connectivity(self, element_id: int) -> np.ndarray:
        """Get connectivity for a specific element ID."""
        idx = np.where(self.ids == element_id)[0]
        if len(idx) == 0:
            raise ValueError(f"Element ID {element_id} not found")
        return self.data[idx[0]]
    
    def update_connectivity(self, element_id: int, connectivity: np.ndarray):
        """Update connectivity for a specific element ID."""
        idx = np.where(self.ids == element_id)[0]
        if len(idx) == 0:
            raise ValueError(f"Element ID {element_id} not found")
        self.data[idx[0]] = connectivity
        self.modified = True


@dataclass
class LightData:
    """
    Base class for configuration-based data (Materials, Boundary Conditions, Steps).
    Uses dictionaries for flexibility.
    """
    name: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    modified: bool = False
    
    def validate(self) -> List[str]:
        """
        Validate the data structure.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.name:
            errors.append("Name is required")
        
        return errors


@dataclass
class Material(LightData):
    """
    Material data structure.
    - name: Material name
    - data: Dictionary containing material properties
      Example: {'elastic': {'E': 200000.0, 'nu': 0.3}}
    """
    
    def validate(self) -> List[str]:
        """Validate material-specific data."""
        errors = super().validate()
        
        # Check for required material properties
        if 'elastic' in self.data:
            elastic = self.data['elastic']
            if 'E' not in elastic:
                errors.append("Elastic modulus (E) is required for elastic materials")
            else:
                E = elastic['E']
                if not isinstance(E, (int, float)) or E <= 0:
                    errors.append("Elastic modulus (E) must be a positive number")
            
            if 'nu' in elastic:
                nu = elastic['nu']
                if not isinstance(nu, (int, float)) or nu < 0 or nu >= 0.5:
                    errors.append("Poisson's ratio (nu) must be between 0 and 0.5")
        
        return errors


@dataclass
class BoundaryCondition(LightData):
    """
    Boundary condition data structure.
    - name: BC name
    - data: Dictionary containing BC properties
      Example: {'set_name': 'Set-1', 'type': 'XSYMM', 'value': None}
    """
    
    def validate(self) -> List[str]:
        """Validate boundary condition-specific data."""
        errors = super().validate()
        
        if 'set_name' not in self.data:
            errors.append("Boundary condition must specify a set_name")
        
        if 'type' not in self.data:
            errors.append("Boundary condition must specify a type")
        
        return errors


@dataclass
class Step(LightData):
    """
    Analysis step data structure.
    - name: Step name
    - data: Dictionary containing step properties
      Example: {'type': 'Static', 'nlgeom': 'YES', 'inc': 1000}
    """
    
    def validate(self) -> List[str]:
        """Validate step-specific data."""
        errors = super().validate()
        
        if 'type' not in self.data:
            errors.append("Step must specify a type")
        
        return errors


@dataclass
class NodeSet(LightData):
    """
    Node set data structure.
    - name: Node set name
    - data: Dictionary containing node IDs or generation info
      Example: {'node_ids': [1, 2, 3]} or {'generate': [1, 439, 1]}
    """
    
    def validate(self) -> List[str]:
        """Validate node set-specific data."""
        errors = super().validate()
        
        if 'node_ids' in self.data:
            node_ids = self.data['node_ids']
            if not isinstance(node_ids, (list, np.ndarray)):
                errors.append("node_ids must be a list or array")
            elif len(node_ids) > 0:
                if not all(isinstance(nid, (int, np.integer)) for nid in node_ids):
                    errors.append("All node IDs must be integers")
                if any(nid <= 0 for nid in node_ids):
                    errors.append("All node IDs must be positive")
        
        if 'generate' in self.data:
            gen = self.data['generate']
            if not isinstance(gen, list) or len(gen) != 3:
                errors.append("generate must be a list of 3 integers [start, end, step]")
        
        return errors


@dataclass
class ElementSet(LightData):
    """
    Element set data structure.
    - name: Element set name
    - data: Dictionary containing element IDs or generation info
    """
    
    def validate(self) -> List[str]:
        """Validate element set-specific data."""
        errors = super().validate()
        
        if 'element_ids' in self.data:
            element_ids = self.data['element_ids']
            if not isinstance(element_ids, (list, np.ndarray)):
                errors.append("element_ids must be a list or array")
            elif len(element_ids) > 0:
                if not all(isinstance(eid, (int, np.integer)) for eid in element_ids):
                    errors.append("All element IDs must be integers")
                if any(eid <= 0 for eid in element_ids):
                    errors.append("All element IDs must be positive")
        
        if 'generate' in self.data:
            gen = self.data['generate']
            if not isinstance(gen, list) or len(gen) != 3:
                errors.append("generate must be a list of 3 integers [start, end, step]")
        
        return errors


def validate_mesh_syntax(value: Any, expected_type: type) -> bool:
    """
    Validate basic mesh data syntax rules.

    Args:
        value: Value to validate
        expected_type: Expected type (int, float, str, etc.)

    Returns:
        True if valid, False otherwise
    """
    if expected_type == int:
        return isinstance(value, (int, np.integer)) and value > 0
    elif expected_type == float:
        return isinstance(value, (int, float, np.floating))
    elif expected_type == str:
        return isinstance(value, str) and len(value) > 0
    else:
        return isinstance(value, expected_type)
