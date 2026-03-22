"""
Common Fixtures for pytest test suite.

This module provides shared fixtures for testing MFEM mesh file
management, morphing, solver functionality, surrogate models,
acquisition functions, and active learning orchestration.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock
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


# ============================================================================
# Surrogate Model Fixtures
# ============================================================================

@pytest.fixture
def surrogate_config():
    """
    Fixture providing default surrogate configuration.

    Returns:
        SurrogateConfig: Default configuration for DeepONet models
    """
    try:
        from meshforge.surrogate.base import SurrogateConfig
        return SurrogateConfig(
            branch_layers=[64, 64],
            trunk_layers=[64, 64],
            activation="tanh",
            learning_rate=1e-3,
            epochs=100,
            batch_size=32,
        )
    except ImportError:
        pytest.skip("meshforge.surrogate not available")


@pytest.fixture
def sample_branch_inputs():
    """
    Fixture providing sample branch network inputs.

    Returns:
        np.ndarray: Shape (n_samples, n_sensors * input_dim)
    """
    n_samples = 10
    n_sensors = 5
    input_dim = 2
    return np.random.randn(n_samples, n_sensors * input_dim).astype(np.float32)


@pytest.fixture
def sample_trunk_inputs():
    """
    Fixture providing sample trunk network inputs (coordinates).

    Returns:
        np.ndarray: Shape (n_points, coord_dim)
    """
    n_points = 50
    coord_dim = 2
    return np.random.randn(n_points, coord_dim).astype(np.float32)


@pytest.fixture
def sample_surrogate_outputs(sample_branch_inputs, sample_trunk_inputs):
    """
    Fixture providing sample outputs for surrogate training.

    Returns:
        np.ndarray: Shape (n_samples, n_points, output_dim)
    """
    n_samples = sample_branch_inputs.shape[0]
    n_points = sample_trunk_inputs.shape[0]
    output_dim = 1
    return np.random.randn(n_samples, n_points, output_dim).astype(np.float32)


# ============================================================================
# Mock Model Fixtures
# ============================================================================

@pytest.fixture
def mock_surrogate_model():
    """
    Fixture providing a mock surrogate model with uncertainty.

    Returns:
        MagicMock: Mock model with predict method
    """
    try:
        from meshforge.surrogate.base import PredictionResult
    except ImportError:
        pytest.skip("meshforge.surrogate not available")

    model = MagicMock()

    def mock_predict(params, coords):
        n_points = coords.shape[0]
        return PredictionResult(
            values=np.random.randn(1, n_points, 1),
            uncertainty=np.abs(np.random.randn(1, n_points, 1)) * 0.1,
            coordinates=coords,
            metadata={}
        )

    model.predict = mock_predict
    model._is_trained = True
    return model


@pytest.fixture
def mock_ensemble_model():
    """
    Fixture providing a mock ensemble model with multiple members.

    Returns:
        MagicMock: Mock ensemble with _models attribute
    """
    try:
        from meshforge.surrogate.base import PredictionResult
    except ImportError:
        pytest.skip("meshforge.surrogate not available")

    model = MagicMock()

    # Create mock ensemble members
    members = []
    for _ in range(5):
        member = MagicMock()
        def member_predict(params, coords):
            n_points = coords.shape[0]
            return PredictionResult(
                values=np.random.randn(1, n_points, 1),
                coordinates=coords,
                metadata={}
            )
        member.predict = member_predict
        members.append(member)

    model._models = members
    model._is_trained = True

    def ensemble_predict(params, coords):
        n_points = coords.shape[0]
        return PredictionResult(
            values=np.random.randn(1, n_points, 1),
            uncertainty=np.abs(np.random.randn(1, n_points, 1)) * 0.1,
            coordinates=coords,
            metadata={"uncertainty_method": "ensemble_std"}
        )

    model.predict = ensemble_predict
    return model


# ============================================================================
# Error Analysis Fixtures
# ============================================================================

@pytest.fixture
def sample_error_field_2d(sample_node_data_2d):
    """
    Fixture providing a sample error field with hotspot.

    Returns:
        np.ndarray: Error values per node
    """
    coords = sample_node_data_2d['coordinates']
    # Error concentrated near (1.5, 0.5)
    hotspot_center = np.array([1.5, 0.5])
    distances = np.linalg.norm(coords - hotspot_center, axis=1)
    return np.exp(-distances)


@pytest.fixture
def sample_grid_coordinates():
    """
    Fixture providing a regular 2D grid of coordinates.

    Returns:
        np.ndarray: Shape (n_points, 2)
    """
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


@pytest.fixture
def sample_true_field(sample_grid_coordinates):
    """
    Fixture providing sample ground truth field values.

    Returns:
        np.ndarray: Field values at grid coordinates
    """
    coords = sample_grid_coordinates
    return np.sin(coords[:, 0]) * np.cos(coords[:, 1])


# ============================================================================
# Active Learning Fixtures
# ============================================================================

@pytest.fixture
def sample_parameter_bounds():
    """
    Fixture providing parameter bounds for active learning.

    Returns:
        dict: Parameter name to (min, max) bounds
    """
    return {
        "delta_R": (-0.5, 0.5),
        "E": (100e9, 300e9),
        "nu": (0.25, 0.35),
    }


@pytest.fixture
def sample_candidate_params(sample_parameter_bounds):
    """
    Fixture providing candidate parameter samples.

    Returns:
        np.ndarray: Shape (n_candidates, n_params)
    """
    n_candidates = 50
    bounds = sample_parameter_bounds
    param_names = list(bounds.keys())

    candidates = np.zeros((n_candidates, len(param_names)))
    for i, name in enumerate(param_names):
        min_val, max_val = bounds[name]
        candidates[:, i] = np.random.uniform(min_val, max_val, n_candidates)

    return candidates


@pytest.fixture
def adaptive_config(tmp_path, beam_quad_mesh_file):
    """
    Fixture providing adaptive orchestrator configuration.

    Returns:
        AdaptiveConfig: Configuration for testing
    """
    try:
        from meshforge.orchestration.adaptive import AdaptiveConfig
    except ImportError:
        pytest.skip("meshforge.orchestration not available")

    output_dir = tmp_path / "adaptive_output"
    output_dir.mkdir(exist_ok=True)

    return AdaptiveConfig(
        base_mesh_path=beam_quad_mesh_file,
        output_dir=output_dir,
        parameter_bounds={"delta_R": (-0.5, 0.5)},
        initial_samples=5,
        max_samples=20,
        convergence_threshold=0.01,
    )


# ============================================================================
# Benchmark Problem Fixtures
# ============================================================================

@pytest.fixture
def plate_with_hole_mesh_content():
    """
    Fixture providing plate with hole mesh content.

    Returns:
        str: MFEM mesh file content
    """
    return """MFEM mesh v1.0

dimension
2

elements
4
1 3 0 1 5 4
1 3 1 2 6 5
1 3 4 5 9 8
1 3 5 6 10 9

boundary
8
1 1 0 1
1 1 1 2
2 1 2 6
2 1 6 10
3 1 10 9
3 1 9 8
4 1 8 4
4 1 4 0

vertices
12
2
0 0
1 0
2 0
0 1
1 1
2 1
0 2
1 2
2 2
0 3
1 3
2 3
"""


@pytest.fixture
def plate_with_hole_mesh_file(tmp_path, plate_with_hole_mesh_content):
    """
    Fixture providing a temporary plate with hole mesh file.

    Returns:
        Path: Path to mesh file
    """
    mesh_file = tmp_path / "plate_with_hole.mesh"
    mesh_file.write_text(plate_with_hole_mesh_content, encoding='utf-8')
    return mesh_file
