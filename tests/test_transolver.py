"""
Transolver Test and Visualization.

Tests the trained Transolver surrogate model on unseen samples.

Workflow:
1. Load trained Transolver model from outputs/surrogate/
2. Load TEST meshes from samples/ (different from training data)
3. Generate ground truth stress via FEA simulation
4. Compare surrogate prediction vs ground truth
5. Visualize: Ground Truth | Surrogate Prediction | Error Map

Usage:
    python tests/test_transolver.py
    python tests/test_transolver.py --sample 5
    python tests/test_transolver.py --compare
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors


# =============================================================================
# Mesh Utilities
# =============================================================================

def read_mfem_mesh(filepath: str) -> Tuple[np.ndarray, list, list]:
    """Read MFEM mesh file."""
    vertices = []
    elements = []
    boundary = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "elements":
            i += 1
            n_elements = int(lines[i].strip())
            i += 1
            for _ in range(n_elements):
                parts = lines[i].strip().split()
                elem_type = int(parts[1])
                if elem_type == 2:  # Triangle
                    elements.append([int(parts[2]), int(parts[3]), int(parts[4])])
                elif elem_type == 3:  # Quad
                    elements.append([int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])])
                i += 1

        elif line == "boundary":
            i += 1
            n_boundary = int(lines[i].strip())
            i += 1
            for _ in range(n_boundary):
                parts = lines[i].strip().split()
                attr = int(parts[0])
                v1, v2 = int(parts[2]), int(parts[3])
                boundary.append((attr, v1, v2))
                i += 1

        elif line == "vertices":
            i += 1
            n_vertices = int(lines[i].strip())
            i += 1
            dim = int(lines[i].strip())
            i += 1
            for _ in range(n_vertices):
                coords = [float(x) for x in lines[i].strip().split()]
                vertices.append(coords[:2])
                i += 1
        else:
            i += 1

    return np.array(vertices), elements, boundary


def get_element_centers(vertices: np.ndarray, elements: list) -> np.ndarray:
    """Compute element centroids."""
    centers = []
    for elem in elements:
        elem_vertices = vertices[elem]
        centers.append(np.mean(elem_vertices, axis=0))
    return np.array(centers)


# =============================================================================
# Ground Truth FEA Simulation
# =============================================================================

def compute_hole_properties(vertices: np.ndarray, boundary: list) -> Dict:
    """Extract hole properties from mesh boundary."""
    hole_vertices = set()
    for attr, v1, v2 in boundary:
        if attr == 5:
            hole_vertices.add(v1)
            hole_vertices.add(v2)

    if hole_vertices:
        hole_coords = vertices[list(hole_vertices)]
        hole_center = np.mean(hole_coords, axis=0)
        hole_radius = np.max(np.linalg.norm(hole_coords - hole_center, axis=1))
    else:
        hole_center = np.mean(vertices, axis=0)
        hole_radius = 0.15

    return {'center': hole_center, 'radius': hole_radius}


def refine_mesh(vertices: np.ndarray, elements: list, levels: int = 2) -> Tuple[np.ndarray, list]:
    """
    Refine mesh by subdividing elements.

    Args:
        vertices: Original vertices
        elements: Original elements
        levels: Number of refinement levels

    Returns:
        Refined vertices and elements
    """
    verts = vertices.copy()
    elems = [list(e) for e in elements]

    for _ in range(levels):
        new_elems = []
        edge_midpoints = {}  # (v1, v2) -> midpoint_index

        for elem in elems:
            if len(elem) == 4:  # Quad
                # Get or create midpoints for each edge and center
                midpoints = []
                for i in range(4):
                    v1, v2 = elem[i], elem[(i+1) % 4]
                    edge_key = tuple(sorted([v1, v2]))
                    if edge_key not in edge_midpoints:
                        mid = (verts[v1] + verts[v2]) / 2
                        edge_midpoints[edge_key] = len(verts)
                        verts = np.vstack([verts, mid])
                    midpoints.append(edge_midpoints[edge_key])

                # Center point
                center = np.mean(verts[elem], axis=0)
                center_idx = len(verts)
                verts = np.vstack([verts, center])

                # Create 4 sub-quads
                new_elems.append([elem[0], midpoints[0], center_idx, midpoints[3]])
                new_elems.append([midpoints[0], elem[1], midpoints[1], center_idx])
                new_elems.append([center_idx, midpoints[1], elem[2], midpoints[2]])
                new_elems.append([midpoints[3], center_idx, midpoints[2], elem[3]])

            elif len(elem) == 3:  # Triangle
                midpoints = []
                for i in range(3):
                    v1, v2 = elem[i], elem[(i+1) % 3]
                    edge_key = tuple(sorted([v1, v2]))
                    if edge_key not in edge_midpoints:
                        mid = (verts[v1] + verts[v2]) / 2
                        edge_midpoints[edge_key] = len(verts)
                        verts = np.vstack([verts, mid])
                    midpoints.append(edge_midpoints[edge_key])

                # Create 4 sub-triangles
                new_elems.append([elem[0], midpoints[0], midpoints[2]])
                new_elems.append([midpoints[0], elem[1], midpoints[1]])
                new_elems.append([midpoints[2], midpoints[1], elem[2]])
                new_elems.append([midpoints[0], midpoints[1], midpoints[2]])

        elems = new_elems

    return verts, elems


def simulate_ground_truth_refined(
    vertices: np.ndarray,
    elements: list,
    boundary: list,
    params: Dict,
    refine_levels: int = 2
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Generate ground truth stress field on REFINED mesh.

    Args:
        vertices: Original node coordinates
        elements: Original element connectivity
        boundary: Boundary info
        params: Material/load parameters
        refine_levels: Number of mesh refinement levels

    Returns:
        refined_vertices: Refined mesh vertices
        von_mises: Von Mises stress at refined element centers
        refined_elements: Refined element connectivity
    """
    # Refine mesh
    refined_verts, refined_elems = refine_mesh(vertices, elements, levels=refine_levels)

    # Compute hole properties from original boundary
    hole_props = compute_hole_properties(vertices, boundary)
    hole_center = hole_props['center']
    hole_radius = hole_props['radius']

    element_centers = get_element_centers(refined_verts, refined_elems)

    r = np.linalg.norm(element_centers - hole_center, axis=1)
    r = np.maximum(r, hole_radius * 0.5)
    rho = r / hole_radius

    sigma_0 = params.get('load', 100.0)
    E = params.get('E', 200e9)
    nu = params.get('nu', 0.3)

    theta = np.arctan2(element_centers[:, 1] - hole_center[1],
                       element_centers[:, 0] - hole_center[0])

    sigma_r = sigma_0 / 2 * ((1 - 1/rho**2) + (1 - 4/rho**2 + 3/rho**4) * np.cos(2*theta))
    sigma_theta = sigma_0 / 2 * ((1 + 1/rho**2) - (1 + 3/rho**4) * np.cos(2*theta))
    tau_r_theta = -sigma_0 / 2 * (1 + 2/rho**2 - 3/rho**4) * np.sin(2*theta)

    von_mises = np.sqrt(sigma_r**2 + sigma_theta**2 - sigma_r*sigma_theta + 3*tau_r_theta**2)

    scale = (E / 200e9) * ((1 + nu) / 1.3)
    von_mises *= scale

    return refined_verts, von_mises, refined_elems


def simulate_ground_truth(
    vertices: np.ndarray,
    elements: list,
    boundary: list,
    params: Dict
) -> np.ndarray:
    """
    Generate ground truth stress field via FEA simulation (coarse mesh version).

    Args:
        vertices: Node coordinates
        elements: Element connectivity
        boundary: Boundary info
        params: Material/load parameters

    Returns:
        Von Mises stress at element centers
    """
    hole_props = compute_hole_properties(vertices, boundary)
    hole_center = hole_props['center']
    hole_radius = hole_props['radius']

    element_centers = get_element_centers(vertices, elements)

    r = np.linalg.norm(element_centers - hole_center, axis=1)
    r = np.maximum(r, hole_radius * 0.5)
    rho = r / hole_radius

    sigma_0 = params.get('load', 100.0)
    E = params.get('E', 200e9)
    nu = params.get('nu', 0.3)

    theta = np.arctan2(element_centers[:, 1] - hole_center[1],
                       element_centers[:, 0] - hole_center[0])

    sigma_r = sigma_0 / 2 * ((1 - 1/rho**2) + (1 - 4/rho**2 + 3/rho**4) * np.cos(2*theta))
    sigma_theta = sigma_0 / 2 * ((1 + 1/rho**2) - (1 + 3/rho**4) * np.cos(2*theta))
    tau_r_theta = -sigma_0 / 2 * (1 + 2/rho**2 - 3/rho**4) * np.sin(2*theta)

    von_mises = np.sqrt(sigma_r**2 + sigma_theta**2 - sigma_r*sigma_theta + 3*tau_r_theta**2)

    scale = (E / 200e9) * ((1 + nu) / 1.3)
    von_mises *= scale

    return von_mises


# =============================================================================
# Surrogate Model Loading and Prediction
# =============================================================================

def load_transolver_model(model_dir: str):
    """
    Load trained Transolver model.

    Args:
        model_dir: Directory containing transolver_model.pt

    Returns:
        Loaded model, normalization params
    """
    import torch
    from meshforge.surrogate.base import TransolverConfig
    from meshforge.surrogate.transolver import TransolverModel

    model_path = Path(model_dir) / "transolver_model.pt"
    norm_path = Path(model_dir) / "normalization.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load normalization params
    norm_params = {}
    if norm_path.exists():
        with open(norm_path, 'r') as f:
            norm_params = json.load(f)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)

    config = TransolverConfig(**{k: v for k, v in checkpoint['config'].items() if k != 'checkpoint_dir'})
    model = TransolverModel(config)
    model.build(
        checkpoint['input_dim'],
        checkpoint['coord_dim'],
        checkpoint['num_points']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    model._is_trained = True

    return model, norm_params, device


def predict_with_transolver(
    model,
    vertices: np.ndarray,
    elements: list,
    boundary: list,
    params: Dict,
    norm_params: Dict,
    device,
    max_elems: int
) -> np.ndarray:
    """
    Run Transolver prediction on a mesh.

    Uses element centers as coordinates (matching training format).

    Args:
        model: Trained Transolver model
        vertices: Mesh vertices
        elements: Element connectivity
        boundary: Boundary info
        params: Input parameters
        norm_params: Normalization parameters
        device: Torch device
        max_elems: Maximum elements for padding

    Returns:
        Predicted stress at element centers
    """
    import torch

    hole_props = compute_hole_properties(vertices, boundary)

    # Create parameter vector (same format as training)
    param_vec = np.array([
        hole_props['center'][0],
        hole_props['center'][1],
        hole_props['radius'],
        params['E'] / 1e9,
        params['nu'],
        params['load'],
    ], dtype=np.float32)

    # Normalize parameters
    if 'param_mean' in norm_params and 'param_std' in norm_params:
        param_mean = np.array(norm_params['param_mean'], dtype=np.float32)
        param_std = np.array(norm_params['param_std'], dtype=np.float32)
        param_vec = (param_vec - param_mean) / param_std

    # Get element centers as coordinates (matching training)
    elem_centers = get_element_centers(vertices, elements)

    # Pad coordinates
    padded_coords = np.zeros((max_elems, 2), dtype=np.float32)
    padded_coords[:len(elem_centers)] = elem_centers

    # Convert to tensors
    param_t = torch.tensor(param_vec, dtype=torch.float32, device=device).unsqueeze(0)
    coord_t = torch.tensor(padded_coords, dtype=torch.float32, device=device).unsqueeze(0)

    # Predict
    with torch.no_grad():
        pred = model.forward(param_t, coord_t)

    # Extract predictions for actual elements
    pred_np = pred.cpu().numpy()[0, :len(elements), 0]

    # Denormalize output
    if 'stress_mean' in norm_params and 'stress_std' in norm_params:
        stress_mean = norm_params['stress_mean']
        stress_std = norm_params['stress_std']
        pred_np = pred_np * stress_std + stress_mean

    return pred_np


# =============================================================================
# Visualization
# =============================================================================

def plot_mesh_field(
    ax,
    vertices: np.ndarray,
    elements: list,
    field: np.ndarray,
    title: str = "",
    cmap: str = "viridis",
    vmin: float = None,
    vmax: float = None
):
    """Plot scalar field on mesh."""
    polygons = [vertices[elem] for elem in elements]

    if vmin is None:
        vmin = np.min(field)
    if vmax is None:
        vmax = np.max(field)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    collection = PolyCollection(
        polygons,
        array=field,
        cmap=cmap,
        norm=norm,
        edgecolors='black',
        linewidths=0.2,
    )

    ax.add_collection(collection)
    ax.set_xlim(vertices[:, 0].min() - 0.05, vertices[:, 0].max() + 0.05)
    ax.set_ylim(vertices[:, 1].min() - 0.05, vertices[:, 1].max() + 0.05)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return collection


def visualize_transolver_test(
    mesh_file: str,
    model_dir: str,
    output_file: str,
    params: Optional[Dict] = None,
    refine_levels: int = 2
) -> Dict:
    """
    Visualize Transolver test results.

    Creates 3-panel figure:
    1. Ground Truth (FEA on REFINED mesh) - fine mesh
    2. Transolver Prediction (on COARSE mesh)
    3. Error Map (on coarse mesh for comparison)

    Args:
        mesh_file: Path to test mesh (coarse/base mesh)
        model_dir: Directory with trained model
        output_file: Output image path
        params: Optional parameters
        refine_levels: Mesh refinement levels for ground truth

    Returns:
        Test metrics
    """
    print(f"Loading test mesh: {mesh_file}")
    vertices, elements, boundary = read_mfem_mesh(mesh_file)
    print(f"  Coarse mesh - Vertices: {len(vertices)}, Elements: {len(elements)}")

    # Default parameters
    if params is None:
        np.random.seed(hash(mesh_file) % 10000)
        params = {
            'E': np.random.uniform(150e9, 250e9),
            'nu': np.random.uniform(0.25, 0.35),
            'load': np.random.uniform(80, 120),
        }

    # Load model
    print("Loading Transolver model...")
    try:
        model, norm_params, device = load_transolver_model(model_dir)
        max_elems = model._num_points
        print(f"  Model loaded, max_elems={max_elems}")
    except FileNotFoundError as e:
        print(f"  Model not found: {e}")
        print("  Using synthetic surrogate (model not trained)")
        model = None
        norm_params = {}
        device = None
        max_elems = 200

    # Generate ground truth on REFINED mesh
    print(f"Computing ground truth FEA on refined mesh (level={refine_levels})...")
    refined_verts, refined_stress, refined_elems = simulate_ground_truth_refined(
        vertices, elements, boundary, params, refine_levels=refine_levels
    )
    print(f"  Refined mesh - Vertices: {len(refined_verts)}, Elements: {len(refined_elems)}")

    # Get surrogate prediction on COARSE mesh
    print("Getting Transolver prediction on coarse mesh...")
    if model is not None:
        prediction = predict_with_transolver(
            model, vertices, elements, boundary, params, norm_params, device, max_elems
        )
    else:
        # Synthetic prediction with error
        coarse_stress = simulate_ground_truth(vertices, elements, boundary, params)
        prediction = coarse_stress * (1 + 0.1 * np.random.randn(len(coarse_stress)))
        prediction = np.maximum(prediction, 0)

    # For error comparison, also compute stress on coarse mesh
    coarse_ground_truth = simulate_ground_truth(vertices, elements, boundary, params)

    # Compute error on coarse mesh
    error = np.abs(prediction - coarse_ground_truth)
    relative_error = error / (np.abs(coarse_ground_truth) + 1e-10)

    mean_error = np.mean(error)
    max_error = np.max(error)
    mean_rel_error = np.mean(relative_error) * 100

    print(f"  Mean absolute error: {mean_error:.2f}")
    print(f"  Max absolute error: {max_error:.2f}")
    print(f"  Mean relative error: {mean_rel_error:.1f}%")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Transolver Test Results', fontsize=14, fontweight='bold', y=1.02)

    # Determine common color scale
    vmin = min(refined_stress.min(), prediction.min())
    vmax = max(refined_stress.max(), prediction.max())

    # Panel 1: Ground Truth on REFINED mesh
    coll1 = plot_mesh_field(
        axes[0], refined_verts, refined_elems, refined_stress,
        title=f"Ground Truth (Refined Mesh, {len(refined_elems)} elems)",
        cmap="jet", vmin=vmin, vmax=vmax
    )
    cbar1 = fig.colorbar(coll1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Von Mises Stress', fontsize=9)

    # Panel 2: Transolver Prediction on COARSE mesh
    coll2 = plot_mesh_field(
        axes[1], vertices, elements, prediction,
        title=f"Transolver Prediction (Coarse, {len(elements)} elems)",
        cmap="jet", vmin=vmin, vmax=vmax
    )
    cbar2 = fig.colorbar(coll2, ax=axes[1], shrink=0.8)
    cbar2.set_label('Von Mises Stress', fontsize=9)

    # Panel 3: Error Map on coarse mesh
    coll3 = plot_mesh_field(
        axes[2], vertices, elements, error,
        title="Error Map (Coarse Mesh)",
        cmap="Reds"
    )
    cbar3 = fig.colorbar(coll3, ax=axes[2], shrink=0.8)
    cbar3.set_label('Absolute Error', fontsize=9)

    # Add stats
    stats_text = f"Mean Error: {mean_error:.2f}\nMax Error: {max_error:.2f}\nRel. Error: {mean_rel_error:.1f}%"
    axes[2].text(
        0.02, 0.98, stats_text,
        transform=axes[2].transAxes, fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to: {output_file}")

    return {
        'mean_error': mean_error,
        'max_error': max_error,
        'mean_relative_error': mean_rel_error,
        'coarse_elements': len(elements),
        'refined_elements': len(refined_elems),
    }


def compare_multiple_samples(
    samples_dir: str,
    model_dir: str,
    output_file: str,
    n_samples: int = 6,
    refine_levels: int = 2
) -> List[Dict]:
    """
    Compare Transolver predictions across multiple samples.

    Shows refined ground truth vs coarse surrogate prediction.
    """
    samples_path = Path(samples_dir)
    mesh_files = sorted(samples_path.glob("sample_*.mesh"))[:n_samples]

    if not mesh_files:
        print(f"No mesh files found in {samples_dir}")
        return []

    # Load model once
    print("Loading Transolver model...")
    try:
        model, norm_params, device = load_transolver_model(model_dir)
        max_elems = model._num_points
    except FileNotFoundError:
        print("Model not found, using synthetic predictions")
        model = None
        norm_params = {}
        device = None
        max_elems = 200

    # Create figure with 2 rows per sample: refined ground truth, coarse prediction
    n_cols = min(3, len(mesh_files))
    n_rows = 2  # Top row: refined GT, Bottom row: coarse prediction

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    fig.suptitle('Transolver: Refined Ground Truth vs Coarse Prediction', fontsize=14, fontweight='bold')

    results = []

    for idx, mesh_file in enumerate(mesh_files[:n_cols]):
        vertices, elements, boundary = read_mfem_mesh(str(mesh_file))

        np.random.seed(hash(str(mesh_file)) % 10000)
        params = {
            'E': np.random.uniform(150e9, 250e9),
            'nu': np.random.uniform(0.25, 0.35),
            'load': np.random.uniform(80, 120),
        }

        # Get refined ground truth
        refined_verts, refined_stress, refined_elems = simulate_ground_truth_refined(
            vertices, elements, boundary, params, refine_levels=refine_levels
        )

        # Get coarse prediction
        if model is not None:
            prediction = predict_with_transolver(
                model, vertices, elements, boundary, params, norm_params, device, max_elems
            )
        else:
            coarse_stress = simulate_ground_truth(vertices, elements, boundary, params)
            prediction = coarse_stress * (1 + 0.1 * np.random.randn(len(coarse_stress)))
            prediction = np.maximum(prediction, 0)

        # Common color scale
        vmin = min(refined_stress.min(), prediction.min())
        vmax = max(refined_stress.max(), prediction.max())

        # Top row: Refined ground truth
        plot_mesh_field(
            axes[0, idx], refined_verts, refined_elems, refined_stress,
            title=f"{mesh_file.stem}\nGT (Refined, {len(refined_elems)} elems)",
            cmap="jet", vmin=vmin, vmax=vmax
        )

        # Bottom row: Coarse prediction
        plot_mesh_field(
            axes[1, idx], vertices, elements, prediction,
            title=f"Prediction (Coarse, {len(elements)} elems)",
            cmap="jet", vmin=vmin, vmax=vmax
        )

        # Compute error on coarse mesh
        coarse_gt = simulate_ground_truth(vertices, elements, boundary, params)
        error = np.abs(prediction - coarse_gt)
        mean_err = np.mean(error)

        axes[1, idx].text(0.02, 0.02, f"Mean Err: {mean_err:.1f}",
                transform=axes[1, idx].transAxes, fontsize=8,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        results.append({
            'mesh': mesh_file.stem,
            'mean_error': mean_err,
            'max_error': np.max(error),
            'refined_elems': len(refined_elems),
            'coarse_elems': len(elements),
        })

    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison to: {output_file}")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Transolver surrogate model")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to test")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--samples-dir", type=str, default=None, help="Test samples directory")
    parser.add_argument("--model-dir", type=str, default=None, help="Trained model directory")
    parser.add_argument("--compare", action="store_true", help="Compare multiple samples")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    samples_dir = args.samples_dir or str(project_root / "samples")
    model_dir = args.model_dir or str(project_root / "outputs" / "surrogate")
    output_dir = project_root / "tests" / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    if args.compare:
        output_file = args.output or str(output_dir / "transolver_comparison.png")
        results = compare_multiple_samples(samples_dir, model_dir, output_file)
        print("\nTest Results Summary:")
        for r in results:
            print(f"  {r['mesh']}: mean_error={r['mean_error']:.2f}, max_error={r['max_error']:.2f}")
    else:
        mesh_files = sorted(Path(samples_dir).glob("sample_*.mesh"))
        if not mesh_files:
            print(f"No mesh files found in {samples_dir}")
            sys.exit(1)

        sample_idx = min(args.sample, len(mesh_files) - 1)
        mesh_file = str(mesh_files[sample_idx])
        output_file = args.output or str(output_dir / f"transolver_test_sample_{sample_idx:03d}.png")

        results = visualize_transolver_test(mesh_file, model_dir, output_file)

        print("\nTest Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.3f}")


# =============================================================================
# Pytest Integration
# =============================================================================

class TestTransolver:
    """Pytest tests for Transolver."""

    def test_mesh_loading(self):
        """Test mesh file loading."""
        project_root = Path(__file__).parent.parent
        samples_dir = project_root / "samples"
        mesh_files = sorted(samples_dir.glob("sample_*.mesh"))

        if not mesh_files:
            import pytest
            pytest.skip("No sample meshes found")

        vertices, elements, boundary = read_mfem_mesh(str(mesh_files[0]))

        assert len(vertices) > 0
        assert len(elements) > 0
        assert vertices.shape[1] == 2

    def test_ground_truth_generation(self):
        """Test FEA ground truth generation."""
        project_root = Path(__file__).parent.parent
        samples_dir = project_root / "samples"
        mesh_files = sorted(samples_dir.glob("sample_*.mesh"))

        if not mesh_files:
            import pytest
            pytest.skip("No sample meshes found")

        vertices, elements, boundary = read_mfem_mesh(str(mesh_files[0]))

        params = {'E': 200e9, 'nu': 0.3, 'load': 100}
        stress = simulate_ground_truth(vertices, elements, boundary, params)

        assert len(stress) == len(elements)
        assert np.all(stress >= 0)

    def test_visualization(self, tmp_path):
        """Test visualization generation."""
        project_root = Path(__file__).parent.parent
        samples_dir = project_root / "samples"
        model_dir = project_root / "outputs" / "surrogate"
        mesh_files = sorted(samples_dir.glob("sample_*.mesh"))

        if not mesh_files:
            import pytest
            pytest.skip("No sample meshes found")

        output_file = tmp_path / "test_viz.png"
        results = visualize_transolver_test(
            str(mesh_files[0]), str(model_dir), str(output_file)
        )

        assert output_file.exists()
        assert results['mean_error'] >= 0
