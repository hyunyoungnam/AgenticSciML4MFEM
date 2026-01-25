"""
Common Fixtures for pytest test suite.

This module provides shared fixtures for testing Abaqus .inp file parsing,
management, and writing functionality.
"""

import pytest
from pathlib import Path


@pytest.fixture
def base_inp_content():
    """
    Fixture that returns a minimal but valid Abaqus .inp string.
    
    Contains:
    - *NODE section with sample nodes
    - *ELEMENT section with sample elements
    - *MATERIAL section with elastic properties
    - *BOUNDARY section with boundary conditions
    - *STEP section with analysis step
    
    Returns:
        str: Valid Abaqus .inp file content
    """
    return """*Heading
** Test Model
*Preprint, echo=NO, model=NO, history=NO, contact=NO
**
** PARTS
**
*Part, name=Part-1
*Node
      1,           0.,           0.
      2,           1.,           0.
      3,           1.,           1.
      4,           0.,           1.
      5,           0.5,          0.5
      6,   1.23456789E-04,   2.34567890E+05
*Element, type=CPS4R
  1,   1,   2,   3,   4
  2,   2,   3,   5,   1
*Nset, nset=Set-1, generate
   1,   4,    1
*Elset, elset=Set-1, generate
   1,   2,    1
** Section: Section-1
*Solid Section, elset=Set-1, material=Material-1
,
*End Part
**  
**
** ASSEMBLY
**
*Assembly, name=Assembly
**  
*Instance, name=Part-1-1, part=Part-1
*End Instance
**  
*Nset, nset=Set-1, instance=Part-1-1
   1,   2,   3,   4
*Nset, nset=Set-2, instance=Part-1-1
   5,   6
*Elset, elset=Set-1, instance=Part-1-1
   1,   2
*End Assembly
** 
** MATERIALS
** 
*Material, name=Material-1
*Elastic
200000., 0.3
** ----------------------------------------------------------------
** 
** STEP: Step-1
** 
*Step, name=Step-1, nlgeom=YES, inc=1000
*Static
0.5, 1., 1e-05, 1.
** 
** BOUNDARY CONDITIONS
** 
** Name: BC-1 Type: Symmetry/Antisymmetry/Encastre
*Boundary
Set-1, XSYMM
** Name: BC-2 Type: Symmetry/Antisymmetry/Encastre
*Boundary
Set-2, YSYMM
** 
** OUTPUT REQUESTS
** 
*Restart, write, frequency=0
** 
** FIELD OUTPUT: F-Output-1
** 
*Output, field, variable=PRESELECT
** 
** HISTORY OUTPUT: H-Output-1
** 
*Output, history, variable=PRESELECT
*End Step
"""


@pytest.fixture
def tmp_inp_file(tmp_path, base_inp_content):
    """
    Fixture that provides a temporary .inp file path with base content.
    
    Uses pytest's tmp_path to create a temporary file that can be used
    for reading and writing during tests. The file is automatically cleaned up
    after the test completes.
    
    Args:
        tmp_path: Pytest fixture providing a temporary directory
        base_inp_content: Fixture providing base .inp content
        
    Returns:
        Path: Path to the temporary .inp file
    """
    inp_file = tmp_path / "test_model.inp"
    inp_file.write_text(base_inp_content, encoding='utf-8')
    return inp_file


@pytest.fixture
def tmp_output_file(tmp_path):
    """
    Fixture that provides a temporary output .inp file path.
    
    Used for writing modified models during tests.
    
    Args:
        tmp_path: Pytest fixture providing a temporary directory
        
    Returns:
        Path: Path to the temporary output .inp file
    """
    return tmp_path / "output_model.inp"


@pytest.fixture
def sample_node_data():
    """
    Fixture providing sample node data for testing.
    
    Returns:
        dict: Dictionary with 'ids' and 'coordinates' keys
    """
    return {
        'ids': [1, 2, 3, 4, 5, 6],
        'coordinates': [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [1.23456789E-04, 2.34567890E+05]
        ]
    }


@pytest.fixture
def sample_element_data():
    """
    Fixture providing sample element data for testing.
    
    Returns:
        dict: Dictionary with 'ids' and 'connectivity' keys
    """
    return {
        'ids': [1, 2],
        'connectivity': [
            [1, 2, 3, 4],
            [2, 3, 5, 1]
        ]
    }


@pytest.fixture
def sample_material_data():
    """
    Fixture providing sample material data for testing.
    
    Returns:
        dict: Dictionary with material properties
    """
    return {
        'name': 'Material-1',
        'elastic': {
            'E': 200000.0,
            'nu': 0.3
        }
    }
