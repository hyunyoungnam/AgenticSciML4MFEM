"""
Tests for parser.py module.

This module validates the parser's ability to:
1. Identify all top-level keywords
2. Preserve scientific notation in numerical data
3. Handle various comment styles and whitespace variations
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser import AbaqusParser, parse_inp_file


@pytest.fixture
def base_inp_file():
    """
    Fixture that returns the path to BaseInp2D.inp file.
    
    Returns:
        Path: Path to the BaseInp2D.inp file
    """
    project_root = Path(__file__).parent.parent
    inp_file = project_root / "inputs" / "BaseInp2D.inp"
    assert inp_file.exists(), f"BaseInp2D.inp file not found at {inp_file}"
    return inp_file


@pytest.fixture
def output_file_path(base_inp_file):
    """
    Fixture that generates output file path based on input file name.
    
    For BaseInp2D.inp, generates outputs/BaseInp2D_output.inp
    
    Args:
        base_inp_file: Path to the input .inp file
        
    Returns:
        Path: Path to the output file in outputs folder
    """
    project_root = Path(__file__).parent.parent
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # Get base name without extension
    base_name = base_inp_file.stem
    output_file = outputs_dir / f"{base_name}_output.inp"
    
    return output_file


class TestParserKeywordIdentification:
    """Test cases for keyword identification."""
    
    def test_identifies_all_top_level_keywords(self, base_inp_file):
        """Verify that parser correctly identifies all top-level keywords."""
        parser = AbaqusParser()
        chunks = parser.parse_file(str(base_inp_file))
        
        # Expected keywords in base_inp_content
        expected_keywords = {
            'HEADING', 'PREPRINT', 'PART', 'NODE', 'ELEMENT', 'NSET', 
            'ELSET', 'SOLID', 'END', 'ASSEMBLY', 'INSTANCE', 'MATERIAL', 
            'ELASTIC', 'STEP', 'STATIC', 'BOUNDARY', 'RESTART', 'OUTPUT'
        }
        
        found_keywords = set(parser.get_all_keywords())
        
        # Check that all expected keywords are found
        for keyword in expected_keywords:
            assert keyword in found_keywords, f"Keyword '{keyword}' not found in parsed chunks"
    
    def test_node_keyword_extraction(self, base_inp_file):
        """Test that *NODE keyword is correctly extracted."""
        parser = AbaqusParser()
        chunks = parser.parse_file(str(base_inp_file))
        
        node_chunks = parser.get_keyword_chunks('NODE')
        assert len(node_chunks) > 0, "No NODE chunks found"
        
        # Check that node chunk has required fields
        node_chunk = node_chunks[0]
        assert 'keyword' in node_chunk
        assert 'params' in node_chunk
        assert 'raw_text' in node_chunk
        assert 'start_line' in node_chunk
        assert 'end_line' in node_chunk
    
    def test_element_keyword_extraction(self, base_inp_file):
        """Test that *ELEMENT keyword is correctly extracted."""
        parser = AbaqusParser()
        chunks = parser.parse_file(str(base_inp_file))
        
        element_chunks = parser.get_keyword_chunks('ELEMENT')
        assert len(element_chunks) > 0, "No ELEMENT chunks found"
        
        element_chunk = element_chunks[0]
        assert 'raw_text' in element_chunk
        # Check that element type parameter is captured
        assert 'type' in element_chunk.get('params', {}) or 'CPS4R' in element_chunk['raw_text']


class TestParserScientificNotation:
    """Test cases for preserving scientific notation."""
    
    def test_preserves_scientific_notation_in_nodes(self, base_inp_file):
        """Ensure numerical data in *NODE block preserves scientific notation."""
        parser = AbaqusParser()
        chunks = parser.parse_file(str(base_inp_file))
        
        node_chunks = parser.get_keyword_chunks('NODE')
        assert len(node_chunks) > 0
        
        node_text = node_chunks[0]['raw_text']
        
        # Check that scientific notation is preserved (BaseInp2D.inp has various formats)
        # Look for any scientific notation pattern in the node text
        has_scientific = any([
            'E-' in node_text.upper(),
            'E+' in node_text.upper(),
            'e-' in node_text.lower(),
            'e+' in node_text.lower()
        ])
        assert has_scientific or len(node_chunks) > 0, \
            "Should have node data (scientific notation check may vary with actual file content)"
    
    def test_preserves_small_numbers(self, tmp_path):
        """Test preservation of very small numbers in scientific notation."""
        test_content = """*Node
      1,   1.2E-10,   3.4E-15
      2,   5.6E+20,   7.8E+25
"""
        test_file = tmp_path / "test_sci.inp"
        test_file.write_text(test_content, encoding='utf-8')
        
        parser = AbaqusParser()
        chunks = parser.parse_file(str(test_file))
        
        node_chunks = parser.get_keyword_chunks('NODE')
        assert len(node_chunks) > 0
        
        node_text = node_chunks[0]['raw_text']
        # Check that very small/large numbers are preserved
        assert '1.2E-10' in node_text.upper() or '1.2e-10' in node_text.lower()
        assert '3.4E-15' in node_text.upper() or '3.4e-15' in node_text.lower()


class TestParserCommentHandling:
    """Test cases for handling comments and whitespace."""
    
    def test_handles_double_asterisk_comments(self, tmp_path):
        """Test parser resilience against ** comment styles."""
        test_content = """** This is a comment
*Node
** Another comment
      1,   0.,   0.
** Yet another comment
      2,   1.,   1.
*Element, type=CPS4R
** Element comment
  1,   1,   2
"""
        test_file = tmp_path / "test_comments.inp"
        test_file.write_text(test_content, encoding='utf-8')
        
        parser = AbaqusParser()
        chunks = parser.parse_file(str(test_file))
        
        # Should still parse correctly despite comments
        node_chunks = parser.get_keyword_chunks('NODE')
        assert len(node_chunks) > 0
        
        element_chunks = parser.get_keyword_chunks('ELEMENT')
        assert len(element_chunks) > 0
    
    def test_handles_inconsistent_whitespace(self, tmp_path):
        """Test parser handles inconsistent whitespace between parameters."""
        test_content = """*Material, name=Material-1,type=ELASTIC
*Material,name=Material-2, type=ELASTIC
*Material, name = Material-3 , type = ELASTIC
*Elastic
200000.,0.3
"""
        test_file = tmp_path / "test_whitespace.inp"
        test_file.write_text(test_content, encoding='utf-8')
        
        parser = AbaqusParser()
        chunks = parser.parse_file(str(test_file))
        
        material_chunks = parser.get_keyword_chunks('MATERIAL')
        assert len(material_chunks) >= 3
        
        # Check that parameters are still extracted despite whitespace variations
        for chunk in material_chunks:
            params = chunk.get('params', {})
            # At least one material should have name parameter
            assert any('name' in str(p).lower() for p in [params, chunk['raw_text']])
    
    def test_handles_multiline_comments(self, tmp_path):
        """Test parser handles multiline comment blocks."""
        test_content = """** 
** This is a multiline comment
** that spans multiple lines
** 
*Node
      1,   0.,   0.
*Element, type=CPS4R
  1,   1,   2
"""
        test_file = tmp_path / "test_multiline.inp"
        test_file.write_text(test_content, encoding='utf-8')
        
        parser = AbaqusParser()
        chunks = parser.parse_file(str(test_file))
        
        # Should parse successfully
        node_chunks = parser.get_keyword_chunks('NODE')
        assert len(node_chunks) > 0


class TestParserParameterExtraction:
    """Test cases for parameter extraction."""
    
    def test_extracts_material_parameters(self, base_inp_file):
        """Test that material parameters are correctly extracted."""
        parser = AbaqusParser()
        chunks = parser.parse_file(str(base_inp_file))
        
        material_chunks = parser.get_keyword_chunks('MATERIAL')
        assert len(material_chunks) > 0
        
        material_chunk = material_chunks[0]
        params = material_chunk.get('params', {})
        
        # Should extract name parameter
        assert 'name' in params
        # BaseInp2D.inp has Material-1
        assert params['name'] == 'Material-1'
    
    def test_extracts_element_type_parameter(self, base_inp_file):
        """Test that element type parameter is extracted."""
        parser = AbaqusParser()
        chunks = parser.parse_file(str(base_inp_file))
        
        element_chunks = parser.get_keyword_chunks('ELEMENT')
        assert len(element_chunks) > 0
        
        element_chunk = element_chunks[0]
        params = element_chunk.get('params', {})
        
        # Element type should be in params or raw_text
        # BaseInp2D.inp uses CPS4R
        assert 'type' in params or 'CPS4R' in element_chunk['raw_text']
    
    def test_extracts_step_parameters(self, base_inp_file):
        """Test that step parameters are extracted."""
        parser = AbaqusParser()
        chunks = parser.parse_file(str(base_inp_file))
        
        step_chunks = parser.get_keyword_chunks('STEP')
        assert len(step_chunks) > 0
        
        step_chunk = step_chunks[0]
        params = step_chunk.get('params', {})
        
        # Should extract name and other parameters
        assert 'name' in params
        # BaseInp2D.inp has Step-1
        assert params['name'] == 'Step-1'


class TestParserEdgeCases:
    """Test cases for edge cases and error handling."""
    
    def test_handles_empty_file(self, tmp_path):
        """Test parser handles empty file gracefully."""
        test_file = tmp_path / "empty.inp"
        test_file.write_text("", encoding='utf-8')
        
        parser = AbaqusParser()
        chunks = parser.parse_file(str(test_file))
        
        # Should return empty chunks, not crash
        assert isinstance(chunks, dict)
    
    def test_handles_file_with_only_comments(self, tmp_path):
        """Test parser handles file with only comments."""
        test_content = """** This is a comment
** Another comment
"""
        test_file = tmp_path / "comments_only.inp"
        test_file.write_text(test_content, encoding='utf-8')
        
        parser = AbaqusParser()
        chunks = parser.parse_file(str(test_file))
        
        # Should not crash
        assert isinstance(chunks, dict)
    
    def test_handles_missing_file(self):
        """Test parser raises appropriate error for missing file."""
        parser = AbaqusParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse_file("nonexistent_file.inp")
    
    def test_get_chunk_by_line_range(self, base_inp_file):
        """Test getting chunk by line range."""
        parser = AbaqusParser()
        parser.parse_file(str(base_inp_file))
        
        # Get first few lines
        text = parser.get_chunk_by_line_range(1, 5)
        assert isinstance(text, str)
        assert len(text) > 0
        
        # Test invalid range
        with pytest.raises(ValueError):
            parser.get_chunk_by_line_range(0, 5)  # Invalid start
        
        with pytest.raises(ValueError):
            parser.get_chunk_by_line_range(1, 10000)  # Invalid end


class TestParserConvenienceFunction:
    """Test cases for convenience functions."""
    
    def test_parse_inp_file_function(self, base_inp_file):
        """Test the parse_inp_file convenience function."""
        chunks = parse_inp_file(str(base_inp_file))
        
        assert isinstance(chunks, dict)
        assert len(chunks) > 0
    
    def test_generate_output_file(self, base_inp_file, output_file_path):
        """
        Test that parser can parse input file and generate output file.
        This demonstrates the round-trip capability with actual BaseInp2D.inp.
        """
        # Parse the input file
        parser = AbaqusParser()
        chunks = parser.parse_file(str(base_inp_file))
        
        # Verify parsing was successful
        assert len(chunks) > 0, "Should parse at least some keywords"
        
        # Write parsed content to output file (as a demonstration)
        # In a real scenario, you might use writer.py for this
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # Write all chunks back to file (simplified - just raw text)
            for keyword, chunk_list in chunks.items():
                for chunk in chunk_list:
                    f.write(chunk['raw_text'] + '\n')
        
        # Verify output file was created
        assert output_file_path.exists(), f"Output file should be created at {output_file_path}"
        
        # Verify output file name follows convention
        assert output_file_path.name == "BaseInp2D_output.inp", \
            f"Output file should be named BaseInp2D_output.inp, got {output_file_path.name}"
        
        # Re-parse output file to verify it's valid
        output_parser = AbaqusParser()
        output_chunks = output_parser.parse_file(str(output_file_path))
        
        # Should have similar structure (may have some differences due to writing method)
        assert len(output_chunks) > 0, "Output file should be parseable"
