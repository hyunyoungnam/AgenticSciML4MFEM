"""
Template-based Reconstructor for Abaqus .inp files.

This module reconstructs an Abaqus .inp file from modified data objects.
For unmodified sections, uses original raw text; for modified sections,
formats data according to Abaqus standard spacing and keyword rules.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from meshforge.parser import AbaqusParser
from meshforge.manager import AbaqusManager
from meshforge.schema import Nodes, Elements, Material, BoundaryCondition, Step, NodeSet, ElementSet

# Standard output file names for the pipeline
OUTPUT_PARSED_INP = "OutputInp2D_parsed.inp"
OUTPUT_MORPHED_INP = "OutputInp2D_morphed.inp"


class AbaqusWriter:
    """
    Writer class that reconstructs .inp files from modified data objects.
    Uses template-based approach: original text for unmodified sections,
    formatted output for modified sections.
    """
    
    # Abaqus formatting constants
    NODE_LINE_WIDTH = 80  # Standard Abaqus line width
    COORD_PRECISION = 8   # Decimal places for coordinates
    ELEMENT_LINE_WIDTH = 80
    
    def __init__(self, manager: AbaqusManager):
        """
        Initialize the writer with a manager instance.
        
        Args:
            manager: AbaqusManager instance with loaded and potentially modified data
        """
        self.manager = manager
        self.parser = manager.parser
        self.output_lines: List[str] = []
    
    def write_file(self, output_path: str) -> str:
        """
        Write the reconstructed .inp file.
        
        Args:
            output_path: Path to write the output file
            
        Returns:
            Path to the written file
        """
        output_path = Path(output_path)
        self.output_lines = []
        
        # Get all chunks in order
        all_chunks = self._get_ordered_chunks()
        
        # Process each chunk
        for chunk_info in all_chunks:
            keyword = chunk_info['keyword']
            chunk = chunk_info['chunk']
            
            if keyword == 'NSET':
                # Strategy A: always generate NSET from schema (mesh is source of truth)
                self._write_nset_from_schema(chunk)
            elif keyword == 'ELSET':
                # Strategy A: always generate ELSET from schema (mesh is source of truth)
                self._write_elset_from_schema(chunk)
            elif keyword in self.manager.modified_sections:
                # Reconstruct modified section
                self._write_modified_section(keyword, chunk)
            else:
                # Use original text
                self._write_original_section(chunk)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.output_lines))
        
        return str(output_path)
    
    def _get_ordered_chunks(self) -> List[Dict]:
        """
        Get all chunks in the order they appear in the file.
        
        Returns:
            List of chunk dictionaries with keyword and chunk info
        """
        chunks = []
        seen_keywords = set()
        
        # Get all unique keywords
        all_keywords = self.parser.get_all_keywords()
        
        # For each keyword, get all its chunks
        for keyword in all_keywords:
            keyword_chunks = self.parser.get_keyword_chunks(keyword)
            for chunk in keyword_chunks:
                chunks.append({
                    'keyword': keyword,
                    'chunk': chunk,
                    'start_line': chunk['start_line']
                })
        
        # Sort by start line to maintain original order
        chunks.sort(key=lambda x: x['start_line'])
        
        return chunks
    
    def _write_original_section(self, chunk: Dict):
        """Write original section text."""
        self.output_lines.append(chunk['raw_text'])
    
    # Strategy A: NSET/ELSET are generated from schema (mesh is source of truth)
    IDS_PER_LINE = 16  # Abaqus-style line wrap for set data
    
    def _write_nset_from_schema(self, chunk: Dict):
        """Write *Nset section from manager.node_sets (generated output)."""
        name = chunk['params'].get('nset', '')
        if not name or name not in self.manager.node_sets:
            self._write_original_section(chunk)
            return
        set_obj = self.manager.node_sets[name]
        # Keyword line: *Nset, nset=name [, generate] [, other params]
        param_parts = [f"nset={name}"]
        if 'generate' in set_obj.data:
            param_parts.append("generate")
        for k, v in chunk['params'].items():
            if k != 'nset' and v:
                param_parts.append(f"{k}={v}")
        self.output_lines.append("*Nset, " + ", ".join(param_parts))
        # Data lines
        if 'generate' in set_obj.data:
            start, end, step = set_obj.data['generate']
            self.output_lines.append(f"   {start},   {end},    {step}")
        else:
            ids = set_obj.data.get('node_ids', [])
            for i in range(0, len(ids), self.IDS_PER_LINE):
                line = ", ".join(f"{x:>5}" for x in ids[i : i + self.IDS_PER_LINE])
                self.output_lines.append(" " + line.strip())
    
    def _write_elset_from_schema(self, chunk: Dict):
        """Write *Elset section from manager.element_sets (generated output)."""
        name = chunk['params'].get('elset', '')
        if not name or name not in self.manager.element_sets:
            self._write_original_section(chunk)
            return
        set_obj = self.manager.element_sets[name]
        # Keyword line: *Elset, elset=name [, generate] [, other params]
        param_parts = [f"elset={name}"]
        if 'generate' in set_obj.data:
            param_parts.append("generate")
        for k, v in chunk['params'].items():
            if k != 'elset' and v:
                param_parts.append(f"{k}={v}")
        self.output_lines.append("*Elset, " + ", ".join(param_parts))
        # Data lines
        if 'generate' in set_obj.data:
            start, end, step = set_obj.data['generate']
            self.output_lines.append(f"   {start},   {end},    {step}")
        else:
            ids = set_obj.data.get('element_ids', [])
            for i in range(0, len(ids), self.IDS_PER_LINE):
                line = ", ".join(f"{x:>5}" for x in ids[i : i + self.IDS_PER_LINE])
                self.output_lines.append(" " + line.strip())
    
    def _write_modified_section(self, keyword: str, chunk: Dict):
        """Write modified section with proper formatting."""
        if keyword == 'NODE' and self.manager.nodes and self.manager.nodes.modified:
            self._write_nodes()
        elif keyword == 'ELEMENT' and self.manager.elements and self.manager.elements.modified:
            self._write_elements()
        elif keyword == 'MATERIAL':
            # Check if this specific material was modified
            material_name = chunk['params'].get('name', '')
            if material_name in self.manager.materials:
                material = self.manager.materials[material_name]
                if material.modified:
                    self._write_material(material, chunk)
                else:
                    self._write_original_section(chunk)
            else:
                self._write_original_section(chunk)
        elif keyword == 'BOUNDARY':
            # Check if this specific BC was modified
            bc_name = chunk.get('name', '')
            # Try to find matching BC
            for bc in self.manager.boundary_conditions.values():
                if bc.modified and bc.data.get('set_name') in chunk['raw_text']:
                    self._write_boundary_condition(bc, chunk)
                    return
            self._write_original_section(chunk)
        else:
            # Unknown modified section, use original
            self._write_original_section(chunk)
    
    def _write_nodes(self):
        """Write node section with proper Abaqus formatting."""
        nodes = self.manager.nodes
        if nodes is None:
            return
        
        # Write keyword line (preserve original format if possible)
        node_chunks = self.parser.get_keyword_chunks('NODE')
        if node_chunks:
            keyword_line = node_chunks[0]['raw_text'].split('\n')[0]
            self.output_lines.append(keyword_line)
        else:
            self.output_lines.append('*Node')
        
        # Write nodes with proper formatting
        for i in range(len(nodes.ids)):
            node_id = nodes.ids[i]
            coords = nodes.data[i]
            
            # Format coordinates
            coord_strs = [f"{coord:.{self.COORD_PRECISION}f}" for coord in coords]
            
            # Abaqus format: ID, X, Y, [Z] with proper spacing
            line = f"{node_id:>10},"
            for coord_str in coord_strs:
                line += f"{coord_str:>16},"
            
            # Remove trailing comma
            line = line.rstrip(',')
            self.output_lines.append(line)
    
    def _write_elements(self):
        """Write element section with proper Abaqus formatting."""
        elements = self.manager.elements
        if elements is None:
            return
        
        # Write keyword line with element type
        element_chunks = self.parser.get_keyword_chunks('ELEMENT')
        if element_chunks:
            keyword_line = element_chunks[0]['raw_text'].split('\n')[0]
            self.output_lines.append(keyword_line)
        else:
            type_str = f", type={elements.element_type}" if elements.element_type else ""
            self.output_lines.append(f'*Element{type_str}')
        
        # Write elements with proper formatting
        for i in range(len(elements.ids)):
            element_id = elements.ids[i]
            connectivity = elements.data[i]
            
            # Format: ID, node1, node2, ...
            line = f"{element_id:>5},"
            for node_id in connectivity:
                line += f"{int(node_id):>5},"
            
            # Remove trailing comma
            line = line.rstrip(',')
            self.output_lines.append(line)
    
    def _write_material(self, material: Material, original_chunk: Dict):
        """Write material section with proper Abaqus formatting."""
        # Write keyword line
        keyword_line = f"*Material, name={material.name}"
        self.output_lines.append(keyword_line)
        
        # Write material properties
        if 'elastic' in material.data:
            self.output_lines.append('*Elastic')
            elastic = material.data['elastic']
            E = elastic.get('E', 0.0)
            nu = elastic.get('nu', 0.0)
            
            # Format: E, nu
            line = f"{E:.6f}, {nu:.6f}"
            self.output_lines.append(line)
        
        # Preserve other material properties from original if they exist
        original_lines = original_chunk['raw_text'].split('\n')
        in_elastic = False
        for line in original_lines[1:]:  # Skip keyword line
            if line.strip().startswith('*Elastic'):
                in_elastic = True
                continue
            elif line.strip().startswith('*'):
                in_elastic = False
                if line.strip().startswith('*Material'):
                    continue
                self.output_lines.append(line)
            elif not in_elastic:
                # This is a property we haven't handled
                self.output_lines.append(line)
    
    def _write_boundary_condition(self, bc: BoundaryCondition, original_chunk: Dict):
        """Write boundary condition section with proper Abaqus formatting."""
        # Write keyword line
        self.output_lines.append('*Boundary')
        
        # Write BC data
        set_name = bc.data.get('set_name', '')
        bc_type = bc.data.get('type', '')
        
        line = f"{set_name}, {bc_type}"
        self.output_lines.append(line)
    
    def format_coordinate(self, coord: float) -> str:
        """
        Format a coordinate value according to Abaqus standards.
        
        Args:
            coord: Coordinate value
            
        Returns:
            Formatted string
        """
        return f"{coord:.{self.COORD_PRECISION}f}"
    
    def format_node_line(self, node_id: int, coords: np.ndarray) -> str:
        """
        Format a single node line according to Abaqus standards.
        
        Args:
            node_id: Node ID
            coords: Coordinate array
            
        Returns:
            Formatted node line string
        """
        line = f"{node_id:>10},"
        for coord in coords:
            line += f"{self.format_coordinate(coord):>16},"
        return line.rstrip(',')
    
    def format_element_line(self, element_id: int, connectivity: np.ndarray) -> str:
        """
        Format a single element line according to Abaqus standards.
        
        Args:
            element_id: Element ID
            connectivity: Connectivity array
            
        Returns:
            Formatted element line string
        """
        line = f"{element_id:>5},"
        for node_id in connectivity:
            line += f"{int(node_id):>5},"
        return line.rstrip(',')


def write_inp_file(manager: AbaqusManager, output_path: str) -> str:
    """
    Convenience function to write an .inp file from a manager.

    Args:
        manager: AbaqusManager instance
        output_path: Path to write the output file

    Returns:
        Path to the written file
    """
    writer = AbaqusWriter(manager)
    return writer.write_file(output_path)


def write_inp_and_vtu(manager: AbaqusManager, output_inp_path: str) -> Tuple[str, str]:
    """
    Write the manager's mesh to both .inp and .vtu (same stem).
    E.g. output_inp_path="outputs/OutputInp2D_morphed.inp" -> writes that and outputs/OutputInp2D_morphed.vtu.

    Args:
        manager: AbaqusManager instance
        output_inp_path: Path for the .inp file (e.g. outputs/OutputInp2D_morphed.inp)

    Returns:
        (inp_path, vtu_path) as strings
    """
    from vtu_export import write_vtu
    path = Path(output_inp_path)
    w = AbaqusWriter(manager)
    inp_path = w.write_file(str(path))
    vtu_path = path.with_suffix(".vtu")
    write_vtu(manager, str(vtu_path))
    return inp_path, str(vtu_path)


def write_parsed_output(manager: AbaqusManager, output_dir: Union[str, Path]) -> Tuple[str, str]:
    """
    Write parsed (non-morphed) mesh to OutputInp2D_parsed.inp and OutputInp2D_parsed.vtu in output_dir.

    Args:
        manager: AbaqusManager instance (e.g. after loading BaseInp2D.inp).
        output_dir: Directory for output files (e.g. outputs/).

    Returns:
        (inp_path, vtu_path) as strings
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inp_path = str(output_dir / OUTPUT_PARSED_INP)
    return write_inp_and_vtu(manager, inp_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
        manager = AbaqusManager(input_file)
        
        # Example modification: scale all node coordinates by 1.1
        if manager.nodes:
            coords = manager.get_nodes()
            if coords is not None:
                manager.update_nodes(coords * 1.1)
        
        writer = AbaqusWriter(manager)
        output_path = writer.write_file(output_file)
        print(f"Written modified .inp file to {output_path}")
    else:
        print("Usage: python writer.py <input_inp_file> <output_inp_file>")
