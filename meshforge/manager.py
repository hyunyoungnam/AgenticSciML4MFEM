"""
Agent API Interface for Abaqus Model Management.

This module serves as the primary API for AI agents (Morphing Agent, Parameter Agent).
It interfaces with parser.py and schema.py, abstracting Abaqus syntax so agents
only interact with Python objects or NumPy arrays.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import re
from pathlib import Path

from meshforge.parser import AbaqusParser
from meshforge.schema import (
    Nodes, Elements, Material, BoundaryCondition, Step,
    NodeSet, ElementSet
)


class AbaqusManager:
    """
    Manager class that provides high-level API for agents to interact with
    Abaqus models without dealing with raw .inp file syntax.
    """
    
    def __init__(self, inp_file_path: str):
        """
        Initialize the manager with an Abaqus .inp file.
        
        Args:
            inp_file_path: Path to the .inp file
        """
        self.inp_file_path = Path(inp_file_path)
        self.parser = AbaqusParser(str(self.inp_file_path))
        
        # Data structures
        self.nodes: Optional[Nodes] = None
        self.elements: Optional[Elements] = None
        self.materials: Dict[str, Material] = {}
        self.boundary_conditions: Dict[str, BoundaryCondition] = {}
        self.steps: Dict[str, Step] = {}
        self.node_sets: Dict[str, NodeSet] = {}
        self.element_sets: Dict[str, ElementSet] = {}
        
        # Track modifications
        self.modified_sections: set = set()
        
        # Load data
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all data from parsed chunks into schema objects."""
        # Load nodes
        node_chunks = self.parser.get_keyword_chunks('NODE')
        if node_chunks:
            self.nodes = self._parse_nodes(node_chunks[0])
        
        # Load elements
        element_chunks = self.parser.get_keyword_chunks('ELEMENT')
        if element_chunks:
            self.elements = self._parse_elements(element_chunks[0])
        
        # Load materials
        material_chunks = self.parser.get_keyword_chunks('MATERIAL')
        for chunk in material_chunks:
            material = self._parse_material(chunk)
            if material:
                self.materials[material.name] = material
        
        # Load boundary conditions
        boundary_chunks = self.parser.get_keyword_chunks('BOUNDARY')
        for chunk in boundary_chunks:
            bc = self._parse_boundary_condition(chunk)
            if bc:
                self.boundary_conditions[bc.name] = bc
        
        # Load steps
        step_chunks = self.parser.get_keyword_chunks('STEP')
        for chunk in step_chunks:
            step = self._parse_step(chunk)
            if step:
                self.steps[step.name] = step
        
        # Load node sets
        nset_chunks = self.parser.get_keyword_chunks('NSET')
        for chunk in nset_chunks:
            nset = self._parse_node_set(chunk)
            if nset:
                self.node_sets[nset.name] = nset
        
        # Load element sets
        elset_chunks = self.parser.get_keyword_chunks('ELSET')
        for chunk in elset_chunks:
            elset = self._parse_element_set(chunk)
            if elset:
                self.element_sets[elset.name] = elset
    
    def _parse_nodes(self, chunk: Dict) -> Nodes:
        """Parse node chunk into Nodes object."""
        lines = chunk['raw_text'].split('\n')[1:]  # Skip keyword line
        node_ids = []
        coordinates = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            
            # Parse node line: ID, X, Y, [Z]
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                try:
                    node_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                    node_ids.append(node_id)
                    coordinates.append(coords)
                except ValueError:
                    continue
        
        nodes = Nodes(
            ids=np.array(node_ids, dtype=np.int32),
            data=np.array(coordinates, dtype=np.float64)
        )
        return nodes
    
    def _parse_elements(self, chunk: Dict) -> Elements:
        """Parse element chunk into Elements object."""
        lines = chunk['raw_text'].split('\n')
        element_type = ""
        element_ids = []
        connectivity_list = []
        
        # Extract element type from keyword line
        keyword_line = lines[0]
        type_match = re.search(r'type=([A-Z0-9]+)', keyword_line, re.IGNORECASE)
        if type_match:
            element_type = type_match.group(1)
        
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            
            # Parse element line: ID, node1, node2, ...
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                try:
                    element_id = int(parts[0])
                    connectivity = [int(p) for p in parts[1:]]
                    element_ids.append(element_id)
                    connectivity_list.append(connectivity)
                except ValueError:
                    continue
        
        elements = Elements(
            ids=np.array(element_ids, dtype=np.int32),
            data=np.array(connectivity_list, dtype=np.int32),
            element_type=element_type
        )
        return elements
    
    def _parse_material(self, chunk: Dict) -> Optional[Material]:
        """Parse material chunk into Material object."""
        name = chunk['params'].get('name', '')
        if not name:
            return None
        
        lines = chunk['raw_text'].split('\n')
        material_data = {}
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('*Elastic'):
                # Parse elastic properties
                if i + 1 < len(lines):
                    props_line = lines[i + 1].strip()
                    props = [float(p.strip()) for p in props_line.split(',')]
                    if len(props) >= 2:
                        material_data['elastic'] = {
                            'E': props[0],
                            'nu': props[1]
                        }
                i += 2
            else:
                i += 1
        
        return Material(name=name, data=material_data)
    
    def _parse_boundary_condition(self, chunk: Dict) -> Optional[BoundaryCondition]:
        """Parse boundary condition chunk into BoundaryCondition object."""
        lines = chunk['raw_text'].split('\n')
        if len(lines) < 2:
            return None
        
        # First line after keyword: set_name, type
        data_line = lines[1].strip()
        parts = [p.strip() for p in data_line.split(',')]
        
        if len(parts) >= 2:
            set_name = parts[0]
            bc_type = parts[1]
            
            bc_data = {
                'set_name': set_name,
                'type': bc_type
            }
            
            # Try to extract name from comment if available
            name = f"BC-{set_name}"
            for line in lines:
                if 'Name:' in line:
                    name_match = re.search(r'Name:\s*([^\s]+)', line)
                    if name_match:
                        name = name_match.group(1)
                    break
            
            return BoundaryCondition(name=name, data=bc_data)
        
        return None
    
    def _parse_step(self, chunk: Dict) -> Optional[Step]:
        """Parse step chunk into Step object."""
        name = chunk['params'].get('name', '')
        if not name:
            return None
        
        step_data = chunk['params'].copy()
        step_data.pop('name', None)
        
        return Step(name=name, data=step_data)
    
    def _parse_node_set(self, chunk: Dict) -> Optional[NodeSet]:
        """Parse node set chunk into NodeSet object."""
        name = chunk['params'].get('nset', '')
        if not name:
            return None
        
        lines = chunk['raw_text'].split('\n')
        nset_data = {}
        
        # Check if it's a generate set
        if 'generate' in chunk['params']:
            # Parse generate line: start, end, step
            if len(lines) > 1:
                gen_line = lines[1].strip()
                parts = [int(p.strip()) for p in gen_line.split(',')]
                if len(parts) == 3:
                    nset_data['generate'] = parts
        else:
            # Parse explicit node IDs
            node_ids = []
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                ids = [int(p.strip()) for p in line.split(',')]
                node_ids.extend(ids)
            nset_data['node_ids'] = node_ids
        
        return NodeSet(name=name, data=nset_data)
    
    def _parse_element_set(self, chunk: Dict) -> Optional[ElementSet]:
        """Parse element set chunk into ElementSet object."""
        name = chunk['params'].get('elset', '')
        if not name:
            return None
        
        lines = chunk['raw_text'].split('\n')
        elset_data = {}
        
        # Check if it's a generate set
        if 'generate' in chunk['params']:
            # Parse generate line: start, end, step
            if len(lines) > 1:
                gen_line = lines[1].strip()
                parts = [int(p.strip()) for p in gen_line.split(',')]
                if len(parts) == 3:
                    elset_data['generate'] = parts
        else:
            # Parse explicit element IDs
            element_ids = []
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                ids = [int(p.strip()) for p in line.split(',')]
                element_ids.extend(ids)
            elset_data['element_ids'] = element_ids
        
        return ElementSet(name=name, data=elset_data)
    
    # Public API methods for agents
    
    def get_nodes(self) -> Optional[np.ndarray]:
        """
        Get all node coordinates as NumPy array.
        
        Returns:
            NumPy array of shape (N, 2) or (N, 3) with node coordinates,
            or None if no nodes loaded
        """
        if self.nodes is None:
            return None
        return self.nodes.data.copy()
    
    def get_node_ids(self) -> Optional[np.ndarray]:
        """
        Get all node IDs as NumPy array.
        
        Returns:
            NumPy array of node IDs, or None if no nodes loaded
        """
        if self.nodes is None:
            return None
        return self.nodes.ids.copy()
    
    def get_node_coordinates(self, node_id: int) -> Optional[np.ndarray]:
        """
        Get coordinates for a specific node.
        
        Args:
            node_id: Node ID
            
        Returns:
            NumPy array of coordinates, or None if node not found
        """
        if self.nodes is None:
            return None
        try:
            return self.nodes.get_coordinates(node_id)
        except ValueError:
            return None
    
    def update_nodes(self, new_coords: np.ndarray, node_ids: Optional[np.ndarray] = None):
        """
        Update node coordinates.
        
        Args:
            new_coords: NumPy array of new coordinates (N, 2) or (N, 3)
            node_ids: Optional array of node IDs to update. If None, updates all nodes
                      in order of existing node IDs.
        """
        if self.nodes is None:
            raise ValueError("No nodes loaded")
        
        if node_ids is None:
            if len(new_coords) != len(self.nodes.ids):
                raise ValueError(f"Number of coordinates ({len(new_coords)}) must match "
                               f"number of nodes ({len(self.nodes.ids)})")
            self.nodes.data = new_coords.copy()
        else:
            if len(new_coords) != len(node_ids):
                raise ValueError(f"Number of coordinates ({len(new_coords)}) must match "
                               f"number of node IDs ({len(node_ids)})")
            for i, node_id in enumerate(node_ids):
                self.nodes.update_coordinates(node_id, new_coords[i])
        
        self.nodes.modified = True
        self.modified_sections.add('NODE')
    
    def get_elements(self) -> Optional[np.ndarray]:
        """
        Get all element connectivity as NumPy array.
        
        Returns:
            NumPy array of element connectivity, or None if no elements loaded
        """
        if self.elements is None:
            return None
        return self.elements.data.copy()
    
    def get_element_ids(self) -> Optional[np.ndarray]:
        """
        Get all element IDs as NumPy array.
        
        Returns:
            NumPy array of element IDs, or None if no elements loaded
        """
        if self.elements is None:
            return None
        return self.elements.ids.copy()
    
    def get_material(self, name: str) -> Optional[Material]:
        """
        Get material by name.
        
        Args:
            name: Material name
            
        Returns:
            Material object or None if not found
        """
        return self.materials.get(name)
    
    def update_material(self, name: str, properties: Dict[str, Any]):
        """
        Update material properties.
        
        Args:
            name: Material name
            properties: Dictionary of material properties to update
        """
        if name not in self.materials:
            raise ValueError(f"Material '{name}' not found")
        
        self.materials[name].data.update(properties)
        self.materials[name].modified = True
        self.modified_sections.add('MATERIAL')
    
    def get_boundary_condition(self, name: str) -> Optional[BoundaryCondition]:
        """
        Get boundary condition by name.
        
        Args:
            name: Boundary condition name
            
        Returns:
            BoundaryCondition object or None if not found
        """
        return self.boundary_conditions.get(name)
    
    def modify_boundary_condition(self, name: str, value: Any, param: str = 'type'):
        """
        Modify a boundary condition parameter.
        
        Args:
            name: Boundary condition name
            value: New value for the parameter
            param: Parameter name to modify (default: 'type')
        """
        if name not in self.boundary_conditions:
            raise ValueError(f"Boundary condition '{name}' not found")
        
        self.boundary_conditions[name].data[param] = value
        self.boundary_conditions[name].modified = True
        self.modified_sections.add('BOUNDARY')
    
    def get_step(self, name: str) -> Optional[Step]:
        """
        Get step by name.
        
        Args:
            name: Step name
            
        Returns:
            Step object or None if not found
        """
        return self.steps.get(name)
    
    def get_node_set(self, name: str) -> Optional[NodeSet]:
        """
        Get node set by name.
        
        Args:
            name: Node set name
            
        Returns:
            NodeSet object or None if not found
        """
        return self.node_sets.get(name)
    
    def get_element_set(self, name: str) -> Optional[ElementSet]:
        """
        Get element set by name.
        
        Args:
            name: Element set name
            
        Returns:
            ElementSet object or None if not found
        """
        return self.element_sets.get(name)
    
    def get_modified_sections(self) -> set:
        """
        Get set of modified section names.
        
        Returns:
            Set of modified section keywords
        """
        return self.modified_sections.copy()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        manager = AbaqusManager(file_path)
        
        print(f"Loaded model from {file_path}")
        print(f"Nodes: {len(manager.nodes.ids) if manager.nodes else 0}")
        print(f"Elements: {len(manager.elements.ids) if manager.elements else 0}")
        print(f"Materials: {len(manager.materials)}")
        print(f"Boundary Conditions: {len(manager.boundary_conditions)}")
        print(f"Steps: {len(manager.steps)}")
    else:
        print("Usage: python manager.py <path_to_inp_file>")
