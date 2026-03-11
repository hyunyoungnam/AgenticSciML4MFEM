"""
Common Fixtures for pytest test suite.

This module provides shared fixtures for testing MFEM mesh file
management, morphing, and solver functionality.
"""

import pytest
from pathlib import Path
import numpy as np


@pytest.fixture
def project_root():
    """Project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def inputs_dir(project_root):
    """Inputs directory containing mesh files."""
    return project_root / "inputs"


@pytest.fixture
def base_mesh_2d_content():
    """
    Fixture that returns a minimal valid MFEM 2D mesh string.

    Contains:
    - 2D quad elements
    - Boundary edges
    - Vertices with 2D coordinates

    Returns:
        str: Valid MFEM mesh file content
    """
    return """MFEM mesh v1.0

dimension
2

elements
6
1 3 0 1 5 4
1 3 1 2 6 5
1 3 2 3 7 6
1 3 4 5 9 8
1 3 5 6 10 9
1 3 6 7 11 10

boundary
8
1 1 0 1
1 1 1 2
1 1 2 3
2 1 3 7
2 1 7 11
3 1 11 10
3 1 10 9
3 1 9 8

vertices
12
2
0 0
1 0
2 0
3 0
0 1
1 1
2 1
3 1
0 2
1 2
2 2
3 2
"""


@pytest.fixture
def base_mesh_3d_content():
    """
    Fixture that returns a minimal valid MFEM 3D mesh string.

    Contains:
    - 3D hex elements
    - Boundary faces
    - Vertices with 3D coordinates

    Returns:
        str: Valid MFEM mesh file content
    """
    return """MFEM mesh v1.0

dimension
3

elements
2
1 5 0 1 3 2 4 5 7 6
1 5 1 8 9 3 5 10 11 7

boundary
10
1 3 0 2 3 1
1 3 4 5 7 6
1 3 0 1 5 4
1 3 2 6 7 3
1 3 0 4 6 2
2 3 1 3 9 8
2 3 5 10 11 7
2 3 1 8 10 5
2 3 3 7 11 9
2 3 8 9 11 10

vertices
12
3
0 0 0
1 0 0
0 1 0
1 1 0
0 0 1
1 0 1
0 1 1
1 1 1
2 0 0
2 1 0
2 0 1
2 1 1
"""


@pytest.fixture
def tmp_mesh_file_2d(tmp_path, base_mesh_2d_content):
    """
    Fixture that provides a temporary 2D .mesh file path with base content.

    Uses pytest's tmp_path to create a temporary file that can be used
    for reading and writing during tests.

    Returns:
        Path: Path to the temporary .mesh file
    """
    mesh_file = tmp_path / "test_model_2d.mesh"
    mesh_file.write_text(base_mesh_2d_content, encoding='utf-8')
    return mesh_file


@pytest.fixture
def tmp_mesh_file_3d(tmp_path, base_mesh_3d_content):
    """
    Fixture that provides a temporary 3D .mesh file path with base content.

    Returns:
        Path: Path to the temporary .mesh file
    """
    mesh_file = tmp_path / "test_model_3d.mesh"
    mesh_file.write_text(base_mesh_3d_content, encoding='utf-8')
    return mesh_file


@pytest.fixture
def tmp_output_mesh(tmp_path):
    """
    Fixture that provides a temporary output .mesh file path.

    Used for writing modified meshes during tests.

    Returns:
        Path: Path to the temporary output .mesh file
    """
    return tmp_path / "output_model.mesh"


@pytest.fixture
def beam_quad_mesh_file(inputs_dir):
    """
    Fixture that returns the path to beam-quad.mesh file.

    This is a simple 2D quad mesh for testing.

    Returns:
        Path: Path to beam-quad.mesh
    """
    mesh_file = inputs_dir / "beam-quad.mesh"
    if not mesh_file.exists():
        pytest.skip(f"beam-quad.mesh not found at {mesh_file}")
    return mesh_file


@pytest.fixture
def beam_hex_mesh_file(inputs_dir):
    """
    Fixture that returns the path to beam-hex.mesh file.

    This is a simple 3D hex mesh for testing.

    Returns:
        Path: Path to beam-hex.mesh
    """
    mesh_file = inputs_dir / "beam-hex.mesh"
    if not mesh_file.exists():
        pytest.skip(f"beam-hex.mesh not found at {mesh_file}")
    return mesh_file


@pytest.fixture
def star_mesh_file(inputs_dir):
    """
    Fixture that returns the path to star.mesh file.

    This is a 2D star-shaped mesh for morphing tests.

    Returns:
        Path: Path to star.mesh
    """
    mesh_file = inputs_dir / "star.mesh"
    if not mesh_file.exists():
        pytest.skip(f"star.mesh not found at {mesh_file}")
    return mesh_file


@pytest.fixture
def sample_node_data_2d():
    """
    Fixture providing sample 2D node data for testing.

    Returns:
        dict: Dictionary with 'ids' and 'coordinates' keys
    """
    return {
        'ids': np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        'coordinates': np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ], dtype=np.float64)
    }


@pytest.fixture
def sample_node_data_3d():
    """
    Fixture providing sample 3D node data for testing.

    Returns:
        dict: Dictionary with 'ids' and 'coordinates' keys
    """
    return {
        'ids': np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
        'coordinates': np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float64)
    }


@pytest.fixture
def sample_element_data_quad():
    """
    Fixture providing sample quad element data for testing.

    Returns:
        dict: Dictionary with 'ids' and 'connectivity' keys
    """
    return {
        'ids': np.array([0, 1], dtype=np.int32),
        'connectivity': np.array([
            [0, 1, 4, 3],
            [1, 2, 5, 4]
        ], dtype=np.int32)
    }


@pytest.fixture
def sample_element_data_hex():
    """
    Fixture providing sample hex element data for testing.

    Returns:
        dict: Dictionary with 'ids' and 'connectivity' keys
    """
    return {
        'ids': np.array([0], dtype=np.int32),
        'connectivity': np.array([
            [0, 1, 3, 2, 4, 5, 7, 6]
        ], dtype=np.int32)
    }


@pytest.fixture
def sample_material_properties():
    """
    Fixture providing sample material properties for testing.

    Returns:
        dict: Dictionary with material properties
    """
    return {
        'E': 200000.0,  # Young's modulus in MPa
        'nu': 0.3,      # Poisson's ratio
        'density': 7850.0,  # kg/m^3
    }
