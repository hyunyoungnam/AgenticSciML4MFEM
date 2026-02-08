"""
Structural Analysis Parser for Abaqus .inp files.

This module performs 'Structural Chunking' by identifying keywords and returning
a dictionary where keys are keywords and values are raw text blocks or line ranges.

Note: While designed to be extensible with Lark grammar, this implementation uses
regex-based parsing for efficiency and simplicity. The structural chunking approach
does not require a full parse tree, making regex suitable for this use case.
"""

import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class AbaqusParser:
    """
    Parser for Abaqus .inp files using structural chunking approach.
    Does not load all data into a single tree; instead scans and returns
    keyword-based chunks.
    """
    
    # Common Abaqus keywords to identify
    KEYWORD_PATTERN = re.compile(
        r'^\*([A-Za-z][A-Za-z0-9_-]*)',  # Matches *KEYWORD (case-insensitive)
        re.MULTILINE
    )
    
    # Pattern for keyword with parameters (e.g., *Material, name=Material-1)
    KEYWORD_WITH_PARAMS = re.compile(
        r'^\*([A-Za-z][A-Za-z0-9_-]*)\s*([^\n]*)',  # Matches *KEYWORD (case-insensitive) and parameters
        re.MULTILINE
    )
    
    # Pattern for parameter extraction (e.g., NAME=Value, TYPE=Material)
    PARAM_PATTERN = re.compile(
        r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,\s]+)',  # Matches PARAM=Value
        re.IGNORECASE
    )
    
    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize the parser.
        
        Args:
            file_path: Optional path to .inp file to parse immediately
        """
        self.file_path = file_path
        self.raw_content = ""
        self.lines = []
        self.chunks: Dict[str, List[Dict]] = {}
        
        if file_path:
            self.parse_file(file_path)
    
    def parse_file(self, file_path: str) -> Dict[str, List[Dict]]:
        """
        Parse an Abaqus .inp file and return structured chunks.
        
        Args:
            file_path: Path to the .inp file
            
        Returns:
            Dictionary where keys are keywords (e.g., 'NODE', 'ELEMENT', 'MATERIAL')
            and values are lists of dictionaries containing:
            - 'keyword': The keyword name
            - 'params': Dictionary of parameters (e.g., {'name': 'Material-1'})
            - 'raw_text': Raw text block for this keyword section
            - 'start_line': Starting line number (1-indexed)
            - 'end_line': Ending line number (1-indexed)
        """
        self.file_path = file_path
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            self.raw_content = f.read()
        
        self.lines = self.raw_content.split('\n')
        self.chunks = self._chunk_file()
        
        return self.chunks
    
    def _chunk_file(self) -> Dict[str, List[Dict]]:
        """
        Perform structural chunking of the file.
        
        Returns:
            Dictionary of keyword chunks
        """
        chunks: Dict[str, List[Dict]] = {}
        current_keyword = None
        current_params = {}
        current_start_line = None
        current_lines = []
        
        i = 0
        while i < len(self.lines):
            line = self.lines[i]
            stripped = line.strip()
            
            # Skip comment lines (Abaqus comments start with **)
            if stripped.startswith('**'):
                i += 1
                continue
            
            # Check if this line is a keyword
            keyword_match = self.KEYWORD_WITH_PARAMS.match(line)
            
            if keyword_match:
                # Save previous chunk if exists
                if current_keyword:
                    chunk_data = {
                        'keyword': current_keyword,
                        'params': current_params.copy(),
                        'raw_text': '\n'.join(current_lines),
                        'start_line': current_start_line,
                        'end_line': i - 1
                    }
                    
                    keyword_base = current_keyword.split(',')[0].strip('*').upper()
                    if keyword_base not in chunks:
                        chunks[keyword_base] = []
                    chunks[keyword_base].append(chunk_data)
                
                # Start new chunk
                keyword_full = keyword_match.group(1)
                param_string = keyword_match.group(2).strip()
                
                current_keyword = keyword_full
                current_params = self._parse_params(param_string)
                current_start_line = i + 1  # 1-indexed
                current_lines = [line]
                
            elif current_keyword:
                # Continue current chunk
                current_lines.append(line)
                
                # Check for end markers (some keywords have explicit end markers)
                if stripped.startswith('*End') and current_keyword:
                    # Check if this is the end of current section
                    end_keyword = stripped.replace('*End', '').strip()
                    if not end_keyword or end_keyword.upper() in current_keyword.upper():
                        current_lines.append(line)
                        # Save chunk
                        chunk_data = {
                            'keyword': current_keyword,
                            'params': current_params.copy(),
                            'raw_text': '\n'.join(current_lines),
                            'start_line': current_start_line,
                            'end_line': i + 1
                        }
                        
                        keyword_base = current_keyword.split(',')[0].strip('*').upper()
                        if keyword_base not in chunks:
                            chunks[keyword_base] = []
                        chunks[keyword_base].append(chunk_data)
                        
                        # Reset
                        current_keyword = None
                        current_params = {}
                        current_start_line = None
                        current_lines = []
                        i += 1
                        continue
            
            i += 1
        
        # Save final chunk if exists
        if current_keyword:
            chunk_data = {
                'keyword': current_keyword,
                'params': current_params.copy(),
                'raw_text': '\n'.join(current_lines),
                'start_line': current_start_line,
                'end_line': len(self.lines)
            }
            
            keyword_base = current_keyword.split(',')[0].strip('*').upper()
            if keyword_base not in chunks:
                chunks[keyword_base] = []
            chunks[keyword_base].append(chunk_data)
        
        return chunks
    
    def _parse_params(self, param_string: str) -> Dict[str, str]:
        """
        Parse parameter string (e.g., "name=Material-1, type=ELASTIC" or "nset=Set-1, generate").
        Captures both key=value pairs and standalone flags (e.g. generate, internal).
        
        Args:
            param_string: String containing parameters
            
        Returns:
            Dictionary of parameter name to value; flags get value '' so "flag" in params works.
        """
        params = {}
        if not param_string:
            return params
        
        for part in param_string.split(','):
            part = part.strip()
            if not part:
                continue
            if '=' in part:
                match = self.PARAM_PATTERN.match(part)
                if match:
                    name, value = match.groups()
                    params[name.lower()] = value.strip('"\'')
            else:
                # Standalone flag (e.g. generate, internal)
                params[part.lower()] = ''
        
        return params
    
    def get_keyword_chunks(self, keyword: str) -> List[Dict]:
        """
        Get all chunks for a specific keyword.
        
        Args:
            keyword: Keyword name (e.g., 'NODE', 'ELEMENT', 'MATERIAL')
            
        Returns:
            List of chunk dictionaries for this keyword
        """
        keyword_upper = keyword.upper().strip('*')
        return self.chunks.get(keyword_upper, [])
    
    def get_all_keywords(self) -> List[str]:
        """
        Get list of all keywords found in the file.
        
        Returns:
            List of keyword names
        """
        return list(self.chunks.keys())
    
    def get_chunk_by_line_range(self, start_line: int, end_line: int) -> str:
        """
        Get raw text for a specific line range.
        
        Args:
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)
            
        Returns:
            Raw text for the specified line range
        """
        if start_line < 1 or end_line > len(self.lines):
            raise ValueError(f"Line range {start_line}-{end_line} out of bounds")
        
        return '\n'.join(self.lines[start_line - 1:end_line])


def parse_inp_file(file_path: str) -> Dict[str, List[Dict]]:
    """
    Convenience function to parse an Abaqus .inp file.
    
    Args:
        file_path: Path to the .inp file
        
    Returns:
        Dictionary of keyword chunks
    """
    parser = AbaqusParser()
    return parser.parse_file(file_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        parser = AbaqusParser()
        chunks = parser.parse_file(file_path)
        
        print(f"Found {len(chunks)} unique keywords:")
        for keyword, chunk_list in chunks.items():
            print(f"  {keyword}: {len(chunk_list)} occurrence(s)")
            for i, chunk in enumerate(chunk_list):
                print(f"    [{i+1}] Lines {chunk['start_line']}-{chunk['end_line']}, "
                      f"Params: {chunk['params']}")
    else:
        print("Usage: python parser.py <path_to_inp_file>")
