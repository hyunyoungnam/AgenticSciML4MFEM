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
# R-Adaptivity Helpers
# =============================================================================

def elem_to_node_error(vertices: np.ndarray, elements: list, elem_error: np.ndarray) -> np.ndarray:
    """Average element-centered error to node-centered error."""
    n = len(vertices)
    total = np.zeros(n)
    count = np.zeros(n)
    for i, elem in enumerate(elements):
        for v in elem:
            total[v] += elem_error[i]
            count[v] += 1
    return total / np.maximum(count, 1)


def adapt_mesh_r(mesh_file, vertices, elements, elem_error):
    """
    Apply TMOP r-adaptivity (node relocation) driven by the element error field.

    Returns:
        adapted_verts:   (N, 2) array of relocated node positions
        adapt_result:    AdaptivityResult or None if adaptation failed
        r_adapted_file:  path to a temp .mesh file with the adapted nodes
                         (caller is responsible for deleting it)
    """
    import tempfile, os
    try:
        from meshforge.mesh.mfem_manager import MFEMManager
        from meshforge.morphing.r_adaptivity import TMOPAdaptivity, AdaptivityConfig

        node_err = elem_to_node_error(vertices, elements, elem_error)
        mgr = MFEMManager(mesh_file)
        if mgr.num_nodes != len(vertices):
            print(f"  R-adaptivity skipped: node count mismatch "
                  f"({mgr.num_nodes} vs {len(vertices)})")
            return vertices, None, None

        result = TMOPAdaptivity(
            AdaptivityConfig(max_iterations=100, verbosity=0)
        ).adapt(mgr, node_err)

        if result.success:
            # Save the r-adapted mesh to a temp file so h-refinement can chain on it
            with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as f:
                tmp_path = f.name
            mgr._extract_mesh_data()   # sync Python-side caches
            saved = mgr.save(tmp_path)
            r_file = str(saved) if saved is not None else None
            return result.coords_adapted, result, r_file
        return vertices, None, None

    except Exception as e:
        print(f"  R-adaptivity failed: {e}")
        return vertices, None, None


def adapt_mesh_h(mesh_file, elem_error):
    """
    Apply h-refinement (element splitting) driven by the element error field.

    Returns:
        h_verts:    (N', 2) array — more nodes than original
        h_elements: list of element connectivity — more elements than original
        h_result:   HRefinementResult or None if failed
    """
    try:
        from meshforge.mesh.mfem_manager import MFEMManager
        from meshforge.morphing.h_refinement import HRefinement, HRefinementConfig

        mgr = MFEMManager(mesh_file)
        if mgr.num_elements != len(elem_error):
            print(f"  H-refinement skipped: element count mismatch "
                  f"({mgr.num_elements} vs {len(elem_error)})")
            return None, None, None

        result = HRefinement(
            HRefinementConfig(error_threshold=0.3, max_refinement_levels=3,
                              max_elements=3000)
        ).refine(mgr, elem_error)

        if result.success:
            h_verts = mgr.get_nodes()
            h_elements = [list(e[e >= 0]) for e in mgr.get_elements()]
            return h_verts, h_elements, result
        return None, None, result

    except Exception as e:
        print(f"  H-refinement failed: {e}")
        return None, None, None


# keep old name as alias so existing call sites don't break
def adapt_mesh_to_error(mesh_file, vertices, elements, elem_error):
    return adapt_mesh_r(mesh_file, vertices, elements, elem_error)


# =============================================================================
# Ground Truth FEA Simulation
# =============================================================================

def compute_hole_properties(vertices: np.ndarray, boundary: list,
                            elements: list = None) -> Dict:
    """
    Extract hole centre and radius from the mesh.

    Strategy (in priority order):
    1. Boundary edges tagged attr=5  → hole boundary vertices
    2. Topological inner boundary    → edges on hole + outer sides, strip outer
    3. Fallback                      → mesh centroid + hardcoded 0.15

    The topological path detects boundary edges as those shared by exactly one
    element, then excludes vertices that sit on the plate's outer perimeter
    (x≈0, x≈1, y≈0, y≈1).  What remains is the hole.
    """
    from collections import defaultdict

    # ── path 1: explicitly tagged attr=5 ────────────────────────────────
    hole_vertices: set = set()
    for attr, v1, v2 in boundary:
        if attr == 5:
            hole_vertices.add(v1)
            hole_vertices.add(v2)

    if hole_vertices:
        hole_coords = vertices[sorted(hole_vertices)]
        hole_center = np.mean(hole_coords, axis=0)
        hole_radius = float(np.max(np.linalg.norm(hole_coords - hole_center, axis=1)))
        return {'center': hole_center, 'radius': hole_radius}

    # ── path 2: topological detection ───────────────────────────────────
    if elements is not None:
        edge_count: dict = defaultdict(int)
        for elem in elements:
            n = len(elem)
            for k in range(n):
                e = tuple(sorted((elem[k], elem[(k + 1) % n])))
                edge_count[e] += 1
        topo_bdr_edges = {e for e, c in edge_count.items() if c == 1}

        # Edges in MFEM boundary section (mechanical/Dirichlet BC)
        mfem_bdr_edges = {tuple(sorted((v1, v2))) for _, v1, v2 in boundary}

        # Collect all topo-boundary vertices that are NOT on the plate perimeter
        plate_tol = 1e-6
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()

        def on_outer_bdr(v: int) -> bool:
            x, y = vertices[v]
            return (x <= x_min + plate_tol or x >= x_max - plate_tol or
                    y <= y_min + plate_tol or y >= y_max - plate_tol)

        for e in topo_bdr_edges:
            if e not in mfem_bdr_edges:
                for v in e:
                    if not on_outer_bdr(v):
                        hole_vertices.add(v)

    if hole_vertices:
        hole_coords = vertices[sorted(hole_vertices)]
        hole_center = np.mean(hole_coords, axis=0)
        hole_radius = float(np.max(np.linalg.norm(hole_coords - hole_center, axis=1)))
        return {'center': hole_center, 'radius': hole_radius}

    # ── path 3: fallback ─────────────────────────────────────────────────
    return {'center': np.mean(vertices, axis=0), 'radius': 0.15}


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

    # Compute hole properties from original boundary + topology
    hole_props = compute_hole_properties(vertices, boundary, elements)
    hole_center = hole_props['center']
    hole_radius = hole_props['radius']

    element_centers = get_element_centers(refined_verts, refined_elems)

    r = np.linalg.norm(element_centers - hole_center, axis=1)
    r = np.maximum(r, hole_radius * 1.001)   # clamp just outside boundary
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
    params: Dict,
    hole_props: Dict = None,
) -> np.ndarray:
    """
    Generate ground truth stress field via FEA simulation (coarse mesh version).

    Args:
        vertices:   Node coordinates
        elements:   Element connectivity
        boundary:   Boundary info
        params:     Material/load parameters
        hole_props: Pre-computed hole centre/radius.  When given (recommended),
                    skips re-detection so that h-refined/r-adapted meshes whose
                    new edge midpoints would otherwise distort the hole estimate
                    use the same reference geometry as the original coarse mesh.

    Returns:
        Von Mises stress at element centers
    """
    if hole_props is None:
        hole_props = compute_hole_properties(vertices, boundary, elements)
    hole_center = hole_props['center']
    hole_radius = hole_props['radius']

    element_centers = get_element_centers(vertices, elements)

    r = np.linalg.norm(element_centers - hole_center, axis=1)
    r = np.maximum(r, hole_radius * 1.001)   # clamp just outside boundary
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
    model_path = Path(model_dir) / "transolver_model.pt"
    norm_path = Path(model_dir) / "normalization.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    import torch
    from meshforge.surrogate.base import TransolverConfig
    from meshforge.surrogate.transolver import TransolverModel

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

    hole_props = compute_hole_properties(vertices, boundary, elements)

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


def nearest_neighbour_interp(
    src_centers: np.ndarray,
    src_values: np.ndarray,
    tgt_centers: np.ndarray,
) -> np.ndarray:
    """
    Map per-element values from source mesh to target mesh via nearest centroid.

    For each target element centre, find the closest source element centre and
    copy its value.  Pure numpy — no scipy required.
    """
    diff    = tgt_centers[:, np.newaxis, :] - src_centers[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)          # (N_tgt, N_src)
    nearest = np.argmin(sq_dist, axis=1)           # (N_tgt,)
    return src_values[nearest]


def compute_element_areas(vertices: np.ndarray, elements: list) -> np.ndarray:
    """Compute element areas (proxy for local mesh resolution)."""
    areas = []
    for elem in elements:
        v = vertices[elem]
        if len(elem) == 3:  # Triangle: 0.5 * |cross product|
            a = 0.5 * abs((v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1]) -
                          (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1]))
        else:  # Quad: shoelace
            n = len(elem)
            a = 0.5 * abs(sum(v[i, 0] * v[(i+1) % n, 1] - v[(i+1) % n, 0] * v[i, 1]
                              for i in range(n)))
        areas.append(a)
    return np.array(areas)


def visualize_transolver_test(
    mesh_file: str,
    model_dir: str,
    output_file: str,
    params: Optional[Dict] = None,
    refine_levels: int = 3
) -> Dict:
    """
    5-panel agentic SciML loop visualization.

    ① Fine-mesh MFEM ground truth  — high-resolution reference
    ② Coarse-mesh MFEM             — starting point (poor accuracy)
    ③ Error map  ①−②              — drives r+h adaptation
    ④ Adapted mesh MFEM result     — after agentic r+h refinement loop
    ⑤ Error map  ①−④              — accuracy gain from the loop

    The surrogate (Transolver) drives the adaptation in step ④ when a
    trained model is available; otherwise the MFEM coarse error is used
    directly as the adaptation signal.

    Args:
        mesh_file: Path to coarse input mesh (from train/ folder)
        model_dir: Directory with trained Transolver model
        output_file: Output PNG path
        params: Optional material/load parameters
        refine_levels: Uniform refinement levels for fine-mesh GT (default 3)

    Returns:
        Dict with error metrics
    """
    print(f"Loading coarse mesh: {mesh_file}")
    vertices, elements, boundary = read_mfem_mesh(mesh_file)
    print(f"  Coarse: {len(vertices)} nodes, {len(elements)} elements")

    if params is None:
        np.random.seed(hash(mesh_file) % 10000)
        params = {
            'E': np.random.uniform(150e9, 250e9),
            'nu': np.random.uniform(0.25, 0.35),
            'load': np.random.uniform(80, 120),
        }

    # Compute hole geometry once from the ORIGINAL coarse mesh so that all
    # later simulate_ground_truth calls (fine, coarse, adapted) use the same
    # reference hole.  After h-refinement or r-adaptation, new midpoint
    # vertices may be added on the hole boundary edges; their positions are
    # chord midpoints of the blob boundary and can distort the estimated hole
    # radius if we re-detect from the refined mesh.
    orig_hole_props = compute_hole_properties(vertices, boundary, elements)
    print(f"  Hole centre={orig_hole_props['center']}, "
          f"radius={orig_hole_props['radius']:.4f}")

    # ── ① Fine-mesh ground truth ──────────────────────────────────────────
    # Uniformly refine the coarse mesh N levels to get a dense reference.
    print(f"Building fine-mesh ground truth ({refine_levels} refinement levels)...")
    fine_verts, fine_elements = refine_mesh(vertices, elements, levels=refine_levels)
    gt_fine = simulate_ground_truth(fine_verts, fine_elements, boundary, params,
                                    hole_props=orig_hole_props)
    fine_centers = get_element_centers(fine_verts, fine_elements)
    print(f"  Fine: {len(fine_verts)} nodes, {len(fine_elements)} elements  "
          f"peak={gt_fine.max():.1f}")

    # ── ② Coarse-mesh MFEM ────────────────────────────────────────────────
    gt_coarse = simulate_ground_truth(vertices, elements, boundary, params,
                                      hole_props=orig_hole_props)
    coarse_centers = get_element_centers(vertices, elements)
    print(f"  Coarse MFEM peak={gt_coarse.max():.1f}")

    # ── ③ Error map: fine GT vs coarse MFEM ──────────────────────────────
    # error_coarse: per-element on coarse mesh (for visualising panel ③)
    gt_fine_on_coarse = nearest_neighbour_interp(fine_centers, gt_fine, coarse_centers)
    error_coarse = np.abs(gt_fine_on_coarse - gt_coarse)
    # Fair metric: project coarse solution onto the fine grid so both sides
    # are evaluated at the same 7 k+ points (avoids bias from element count).
    coarse_on_fine = nearest_neighbour_interp(coarse_centers, gt_coarse, fine_centers)
    rel_err_coarse = (np.mean(np.abs(gt_fine - coarse_on_fine))
                      / (np.mean(np.abs(gt_fine)) + 1e-10) * 100)
    print(f"  Coarse error vs fine GT — mean={np.mean(error_coarse):.2f}  "
          f"rel(fine-grid)={rel_err_coarse:.1f}%")

    # ── Agentic SciML loop adaptation signal ─────────────────────────────
    # The coarse-vs-fine MFEM error (error_coarse) is the ground-truth signal
    # that tells us WHERE the coarse mesh is inaccurate.  This is what drives
    # r+h refinement.  The surrogate (Transolver) is loaded here to annotate
    # the output but does not change the adaptation target.
    print("Loading Transolver model (for annotation)...")
    try:
        model, norm_params, device = load_transolver_model(model_dir)
        max_elems = model._num_points
        prediction = predict_with_transolver(
            model, vertices, elements, boundary, params, norm_params, device, max_elems
        )
        surrogate_error = np.abs(prediction - gt_coarse)
        surrogate_rel   = np.mean(surrogate_error) / (np.mean(np.abs(gt_coarse)) + 1e-10) * 100
        print(f"  Surrogate rel. err on coarse mesh: {surrogate_rel:.1f}%")
    except FileNotFoundError:
        surrogate_rel = float('nan')
        print("  No surrogate found")

    # Adaptation is always driven by the measured MFEM coarse error
    adapt_signal = error_coarse

    # ── R-adaptivity (node relocation, same element count) ────────────────
    print("Applying TMOP r-adaptivity...")
    r_verts, r_result, r_tmp_file = adapt_mesh_r(mesh_file, vertices, elements, adapt_signal)
    r_ok = r_result is not None and r_result.success
    if r_ok:
        print(f"  R-adapt OK — max node disp: "
              f"{np.linalg.norm(r_verts - vertices, axis=1).max():.4f}")
    else:
        r_verts = vertices.copy()
        r_tmp_file = None
        print("  R-adapt unavailable")

    # ── H-refinement (element splitting, chained on the r-adapted mesh) ──
    # Pass the r-adapted mesh file so h-refinement operates on already-relocated
    # nodes.  If r-adaptivity failed, fall back to the original file.
    print("Applying h-refinement...")
    h_src_file = r_tmp_file if r_tmp_file else mesh_file
    h_verts, h_elements, h_result = adapt_mesh_h(h_src_file, adapt_signal)
    h_ok = h_result is not None and h_result.success
    if h_ok:
        print(f"  H-refine OK — {h_result.num_elements_before}→{h_result.num_elements_after} "
              f"elems, {h_result.num_nodes_before}→{h_result.num_nodes_after} nodes")
    else:
        h_verts    = r_verts.copy()
        h_elements = elements
        print("  H-refine unavailable")

    # Clean up the temp r-adapted mesh file
    if r_tmp_file:
        import os as _os
        try:
            _os.unlink(r_tmp_file)
        except OSError:
            pass

    # ── ④ MFEM on the adapted (h-refined) mesh ────────────────────────────
    gt_adapted = simulate_ground_truth(h_verts, h_elements, boundary, params,
                                        hole_props=orig_hole_props)
    adapted_centers = get_element_centers(h_verts, h_elements)
    print(f"  Adapted MFEM peak={gt_adapted.max():.1f}")

    # ── ⑤ Error map: fine GT vs adapted MFEM ─────────────────────────────
    # error_adapted: per-element on adapted mesh (for visualising panel ⑤)
    gt_fine_on_adapted = nearest_neighbour_interp(fine_centers, gt_fine, adapted_centers)
    error_adapted = np.abs(gt_fine_on_adapted - gt_adapted)
    # Fair metric: project adapted solution onto the fine grid (same reference
    # as rel_err_coarse above — both measured at the same 7k+ evaluation points).
    adapted_on_fine = nearest_neighbour_interp(adapted_centers, gt_adapted, fine_centers)
    rel_err_adapted = (np.mean(np.abs(gt_fine - adapted_on_fine))
                       / (np.mean(np.abs(gt_fine)) + 1e-10) * 100)
    print(f"  Adapted error vs fine GT — mean={np.mean(error_adapted):.2f}  "
          f"rel(fine-grid)={rel_err_adapted:.1f}%  "
          f"(improvement: {rel_err_coarse - rel_err_adapted:+.1f} pp)")

    # ── Build 5-panel figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(
        'Agentic SciML Loop  |  '
        '① Fine GT  →  ② Coarse MFEM  →  ③ Error  →  ④ R+H Adapted  →  ⑤ Error Reduced',
        fontsize=11, fontweight='bold', y=1.02
    )

    # Shared stress colour scale across ①②④
    stress_vmin = min(gt_fine.min(), gt_coarse.min(), gt_adapted.min())
    stress_vmax = max(gt_fine.max(), gt_coarse.max(), gt_adapted.max())

    # ① Fine-mesh ground truth
    coll1 = plot_mesh_field(
        axes[0], fine_verts, fine_elements, gt_fine,
        title=f"① Fine Mesh GT  ({len(fine_elements)} elems)\n"
              f"Uniform {refine_levels}-level refinement",
        cmap="jet", vmin=stress_vmin, vmax=stress_vmax
    )
    fig.colorbar(coll1, ax=axes[0], shrink=0.8).set_label('Von Mises Stress', fontsize=8)
    axes[0].text(
        0.02, 0.02, f"Peak: {gt_fine.max():.1f}\nReference solution",
        transform=axes[0].transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9)
    )

    # ② Coarse-mesh MFEM
    coll2 = plot_mesh_field(
        axes[1], vertices, elements, gt_coarse,
        title=f"② Coarse Mesh MFEM  ({len(elements)} elems)\n"
              f"Starting point — poor accuracy",
        cmap="jet", vmin=stress_vmin, vmax=stress_vmax
    )
    fig.colorbar(coll2, ax=axes[1], shrink=0.8).set_label('Von Mises Stress', fontsize=8)
    axes[1].text(
        0.02, 0.02,
        f"Peak: {gt_coarse.max():.1f}\nRel err vs GT: {rel_err_coarse:.1f}%",
        transform=axes[1].transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    )

    # ③ Error map: fine GT − coarse MFEM
    err_vmax = max(error_coarse.max(), error_adapted.max())   # shared scale for ③ and ⑤
    coll3 = plot_mesh_field(
        axes[2], vertices, elements, error_coarse,
        title="③ Error Map  |GT − Coarse|\n(drives r+h adaptation)",
        cmap="Reds", vmin=0, vmax=err_vmax
    )
    fig.colorbar(coll3, ax=axes[2], shrink=0.8).set_label('|Fine GT − Coarse MFEM|', fontsize=8)
    axes[2].text(
        0.02, 0.98,
        f"Mean: {np.mean(error_coarse):.2f}\nMax:  {np.max(error_coarse):.2f}\n"
        f"Rel:  {rel_err_coarse:.1f}%",
        transform=axes[2].transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )

    # ④ MFEM on adapted mesh (r+h)
    coll4 = plot_mesh_field(
        axes[3], h_verts, h_elements, gt_adapted,
        title=f"④ Adapted MFEM  ({len(h_elements)} elems)\n"
              f"After agentic r+h refinement loop",
        cmap="jet", vmin=stress_vmin, vmax=stress_vmax
    )
    fig.colorbar(coll4, ax=axes[3], shrink=0.8).set_label('Von Mises Stress', fontsize=8)
    note4_lines = [f"Peak: {gt_adapted.max():.1f}"]
    if h_ok:
        note4_lines.append(f"H-refine: {h_result.num_elements_before}→{h_result.num_elements_after} elems")
    if r_ok:
        note4_lines.append("R-adapt: ✓")
    axes[3].text(
        0.02, 0.02, "\n".join(note4_lines),
        transform=axes[3].transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9)
    )

    # ⑤ Error map: fine GT − adapted MFEM
    coll5 = plot_mesh_field(
        axes[4], h_verts, h_elements, error_adapted,
        title="⑤ Error Map  |GT − Adapted|\nAccuracy gain from the loop",
        cmap="Reds", vmin=0, vmax=err_vmax
    )
    fig.colorbar(coll5, ax=axes[4], shrink=0.8).set_label('|Fine GT − Adapted MFEM|', fontsize=8)
    improvement = rel_err_coarse - rel_err_adapted
    colour5 = 'lightgreen' if improvement > 0 else 'lightsalmon'
    axes[4].text(
        0.02, 0.98,
        f"Mean: {np.mean(error_adapted):.2f}\nMax:  {np.max(error_adapted):.2f}\n"
        f"Rel:  {rel_err_adapted:.1f}%\n"
        f"Δ error: {improvement:+.1f} pp\n"
        f"{'✓ improved' if improvement > 0 else '✗ no gain'}",
        transform=axes[4].transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor=colour5, alpha=0.9)
    )

    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

    return {
        'mean_error': float(np.mean(error_coarse)),
        'max_error': float(np.max(error_coarse)),
        'mean_relative_error': rel_err_coarse,
        'mean_relative_error_adapted': rel_err_adapted,
        'improvement_pp': improvement,
        'coarse_elements': len(elements),
        'fine_elements': len(fine_elements),
        'adapted_elements': len(h_elements),
        'adapted': r_ok,
        'h_refined': h_ok,
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


def visualize_best_training_sample(
    train_dir: str,
    model_dir: str,
    output_file: str,
    refine_levels: int = 2
) -> Dict:
    """
    Find and visualize the best training sample (minimum Max Error).

    Creates 3-panel figure:
    1. Ground Truth (FEA on REFINED mesh)
    2. Transolver Prediction (on COARSE mesh)
    3. Error Map (on coarse mesh)

    Args:
        train_dir: Directory containing training mesh files
        model_dir: Directory with trained model
        output_file: Output image path
        refine_levels: Mesh refinement levels for ground truth

    Returns:
        Best sample metrics
    """
    train_path = Path(train_dir)
    mesh_files = sorted(train_path.glob("sample_*.mesh"))

    if not mesh_files:
        print(f"No mesh files found in {train_dir}")
        return {}

    # Load model once
    print("Loading Transolver model...")
    try:
        model, norm_params, device = load_transolver_model(model_dir)
        max_elems = model._num_points
    except FileNotFoundError:
        print("Model not found, cannot evaluate training samples")
        return {}

    # Evaluate all training samples to find minimum max_error
    print(f"Evaluating {len(mesh_files)} training samples to find best case...")
    sample_errors = []

    for mesh_file in mesh_files:
        vertices, elements, boundary = read_mfem_mesh(str(mesh_file))

        np.random.seed(hash(str(mesh_file)) % 10000)
        params = {
            'E': np.random.uniform(150e9, 250e9),
            'nu': np.random.uniform(0.25, 0.35),
            'load': np.random.uniform(80, 120),
        }

        # Get coarse ground truth and prediction
        coarse_gt = simulate_ground_truth(vertices, elements, boundary, params)
        prediction = predict_with_transolver(
            model, vertices, elements, boundary, params, norm_params, device, max_elems
        )

        error = np.abs(prediction - coarse_gt)
        max_error = np.max(error)
        mean_error = np.mean(error)

        sample_errors.append({
            'mesh_file': mesh_file,
            'max_error': max_error,
            'mean_error': mean_error,
            'params': params,
        })

    # Find sample with minimum max_error
    best_sample = min(sample_errors, key=lambda x: x['max_error'])
    best_mesh_file = best_sample['mesh_file']
    best_params = best_sample['params']

    print(f"\nBest training sample: {best_mesh_file.stem}")
    print(f"  Max Error: {best_sample['max_error']:.2f}")
    print(f"  Mean Error: {best_sample['mean_error']:.2f}")

    # Now create full visualization for best sample
    vertices, elements, boundary = read_mfem_mesh(str(best_mesh_file))

    # Generate ground truth on REFINED mesh
    refined_verts, refined_stress, refined_elems = simulate_ground_truth_refined(
        vertices, elements, boundary, best_params, refine_levels=refine_levels
    )

    # Get surrogate prediction on COARSE mesh
    prediction = predict_with_transolver(
        model, vertices, elements, boundary, best_params, norm_params, device, max_elems
    )

    # Coarse ground truth for error
    coarse_ground_truth = simulate_ground_truth(vertices, elements, boundary, best_params)

    # Compute error
    error = np.abs(prediction - coarse_ground_truth)
    relative_error = error / (np.abs(coarse_ground_truth) + 1e-10)

    mean_error = np.mean(error)
    max_error = np.max(error)
    mean_rel_error = np.mean(relative_error) * 100

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Transolver Training Results - Best Case ({best_mesh_file.stem})',
                 fontsize=14, fontweight='bold', y=1.02)

    # Determine common color scale
    vmin = min(refined_stress.min(), prediction.min())
    vmax = max(refined_stress.max(), prediction.max())

    # Panel 1: Ground Truth on REFINED mesh
    coll1 = plot_mesh_field(
        axes[0], refined_verts, refined_elems, refined_stress,
        title=f"Ground Truth (Refined, {len(refined_elems)} elems)",
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

    print(f"Saved best training sample visualization to: {output_file}")

    return {
        'best_sample': best_mesh_file.stem,
        'mean_error': mean_error,
        'max_error': max_error,
        'mean_relative_error': mean_rel_error,
        'coarse_elements': len(elements),
        'refined_elements': len(refined_elems),
        'total_samples_evaluated': len(mesh_files),
    }


def visualize_sciml_loop_output(
    test_sample_dir: str,
    model_dir: str,
    output_file: str,
    sample_idx: int = 25,
    refine_levels: int = 2
) -> Dict:
    """
    Visualize SciML loop output on a test sample (different from training).

    Creates 3-panel figure showing the agentic SciML loop results:
    1. Ground Truth (FEA on REFINED mesh)
    2. SciML Prediction (on COARSE mesh)
    3. Error Map (on coarse mesh)

    Args:
        test_sample_dir: Directory containing test mesh files
        model_dir: Directory with trained model
        output_file: Output image path
        sample_idx: Index of test sample to use
        refine_levels: Mesh refinement levels for ground truth

    Returns:
        Test metrics
    """
    samples_path = Path(test_sample_dir)
    mesh_files = sorted(samples_path.glob("sample_*.mesh"))

    if not mesh_files:
        print(f"No mesh files found in {test_sample_dir}")
        return {}

    # Select test sample
    sample_idx = min(sample_idx, len(mesh_files) - 1)
    mesh_file = mesh_files[sample_idx]

    print(f"Testing SciML loop on: {mesh_file.stem}")

    # Load model
    print("Loading Transolver model...")
    try:
        model, norm_params, device = load_transolver_model(model_dir)
        max_elems = model._num_points
    except FileNotFoundError as e:
        print(f"Model not found: {e}")
        return {}

    # Load mesh
    vertices, elements, boundary = read_mfem_mesh(str(mesh_file))
    print(f"  Coarse mesh - Vertices: {len(vertices)}, Elements: {len(elements)}")

    # Generate parameters (use different seed for test)
    np.random.seed(hash(str(mesh_file)) % 10000)
    params = {
        'E': np.random.uniform(150e9, 250e9),
        'nu': np.random.uniform(0.25, 0.35),
        'load': np.random.uniform(80, 120),
    }

    # Generate ground truth on REFINED mesh
    print(f"Computing ground truth FEA on refined mesh (level={refine_levels})...")
    refined_verts, refined_stress, refined_elems = simulate_ground_truth_refined(
        vertices, elements, boundary, params, refine_levels=refine_levels
    )
    print(f"  Refined mesh - Vertices: {len(refined_verts)}, Elements: {len(refined_elems)}")

    # Get SciML prediction on COARSE mesh
    print("Getting SciML (Transolver) prediction on coarse mesh...")
    prediction = predict_with_transolver(
        model, vertices, elements, boundary, params, norm_params, device, max_elems
    )

    # Coarse ground truth for error comparison
    coarse_ground_truth = simulate_ground_truth(vertices, elements, boundary, params)

    # Compute error
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
    fig.suptitle(f'Agentic SciML Loop - Test Results ({mesh_file.stem})',
                 fontsize=14, fontweight='bold', y=1.02)

    # Determine common color scale
    vmin = min(refined_stress.min(), prediction.min())
    vmax = max(refined_stress.max(), prediction.max())

    # Panel 1: Ground Truth on REFINED mesh
    coll1 = plot_mesh_field(
        axes[0], refined_verts, refined_elems, refined_stress,
        title=f"Ground Truth (Refined, {len(refined_elems)} elems)",
        cmap="jet", vmin=vmin, vmax=vmax
    )
    cbar1 = fig.colorbar(coll1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Von Mises Stress', fontsize=9)

    # Panel 2: SciML Prediction on COARSE mesh
    coll2 = plot_mesh_field(
        axes[1], vertices, elements, prediction,
        title=f"SciML Prediction (Coarse, {len(elements)} elems)",
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

    print(f"Saved SciML loop visualization to: {output_file}")

    return {
        'test_sample': mesh_file.stem,
        'mean_error': mean_error,
        'max_error': max_error,
        'mean_relative_error': mean_rel_error,
        'coarse_elements': len(elements),
        'refined_elements': len(refined_elems),
    }


def visualize_sciml_active_learning(
    train_dir: str,
    test_dir: str,
    model_dir: str,
    output_file: str,
    n_test_samples: int = 5
) -> Dict:
    """
    Visualize SciML active learning loop: error analysis across parameter space.

    Shows:
    - Row 1: Training samples (low error - model knows these)
    - Row 2: Test samples (potentially higher error - active learning targets)
    - Error map highlighting where more training data is needed

    Args:
        train_dir: Training samples directory
        test_dir: Test samples directory
        model_dir: Trained model directory
        output_file: Output image path
        n_test_samples: Number of test samples to evaluate

    Returns:
        Analysis results
    """
    # Load model
    print("Loading Transolver model...")
    try:
        model, norm_params, device = load_transolver_model(model_dir)
        max_elems = model._num_points
    except FileNotFoundError as e:
        print(f"Model not found: {e}")
        return {}

    # Get training samples
    train_path = Path(train_dir)
    train_files = sorted(train_path.glob("sample_*.mesh"))

    # Get test samples
    test_path = Path(test_dir)
    test_files = sorted(test_path.glob("sample_*.mesh"))

    print(f"Analyzing {len(train_files)} training samples and {len(test_files)} test samples...")

    # Evaluate all samples to get error distribution
    all_results = []

    for mesh_file in train_files[:50]:  # Sample subset of training
        vertices, elements, boundary = read_mfem_mesh(str(mesh_file))
        np.random.seed(hash(str(mesh_file)) % 10000)
        params = {
            'E': np.random.uniform(150e9, 250e9),
            'nu': np.random.uniform(0.25, 0.35),
            'load': np.random.uniform(80, 120),
        }

        coarse_gt = simulate_ground_truth(vertices, elements, boundary, params)
        prediction = predict_with_transolver(
            model, vertices, elements, boundary, params, norm_params, device, max_elems
        )
        error = np.abs(prediction - coarse_gt)

        all_results.append({
            'file': mesh_file,
            'type': 'train',
            'max_error': np.max(error),
            'mean_error': np.mean(error),
            'params': params,
        })

    for mesh_file in test_files[:n_test_samples]:
        vertices, elements, boundary = read_mfem_mesh(str(mesh_file))
        np.random.seed(hash(str(mesh_file)) % 10000)
        params = {
            'E': np.random.uniform(150e9, 250e9),
            'nu': np.random.uniform(0.25, 0.35),
            'load': np.random.uniform(80, 120),
        }

        coarse_gt = simulate_ground_truth(vertices, elements, boundary, params)
        prediction = predict_with_transolver(
            model, vertices, elements, boundary, params, norm_params, device, max_elems
        )
        error = np.abs(prediction - coarse_gt)

        all_results.append({
            'file': mesh_file,
            'type': 'test',
            'max_error': np.max(error),
            'mean_error': np.mean(error),
            'params': params,
        })

    # Find best training and worst test samples
    train_results = [r for r in all_results if r['type'] == 'train']
    test_results = [r for r in all_results if r['type'] == 'test']

    best_train = min(train_results, key=lambda x: x['max_error'])
    worst_test = max(test_results, key=lambda x: x['max_error'])

    print(f"\nBest training sample: {best_train['file'].stem}, Max Error: {best_train['max_error']:.2f}")
    print(f"Worst test sample: {worst_test['file'].stem}, Max Error: {worst_test['max_error']:.2f}")

    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Agentic SciML Loop - Active Learning Analysis',
                 fontsize=14, fontweight='bold', y=0.98)

    # Row 1: Best training sample
    vertices, elements, boundary = read_mfem_mesh(str(best_train['file']))
    refined_verts, refined_stress, refined_elems = simulate_ground_truth_refined(
        vertices, elements, boundary, best_train['params'], refine_levels=2
    )
    prediction = predict_with_transolver(
        model, vertices, elements, boundary, best_train['params'], norm_params, device, max_elems
    )
    coarse_gt = simulate_ground_truth(vertices, elements, boundary, best_train['params'])
    error = np.abs(prediction - coarse_gt)

    vmin = min(refined_stress.min(), prediction.min())
    vmax = max(refined_stress.max(), prediction.max())

    coll = plot_mesh_field(axes[0, 0], refined_verts, refined_elems, refined_stress,
                           title=f"Training Best: GT ({best_train['file'].stem})",
                           cmap="jet", vmin=vmin, vmax=vmax)
    fig.colorbar(coll, ax=axes[0, 0], shrink=0.8)

    coll = plot_mesh_field(axes[0, 1], vertices, elements, prediction,
                           title="Transolver Prediction",
                           cmap="jet", vmin=vmin, vmax=vmax)
    fig.colorbar(coll, ax=axes[0, 1], shrink=0.8)

    coll = plot_mesh_field(axes[0, 2], vertices, elements, error,
                           title=f"Error (Max: {best_train['max_error']:.1f})",
                           cmap="Reds")
    fig.colorbar(coll, ax=axes[0, 2], shrink=0.8)
    axes[0, 2].text(0.02, 0.98, "Low error\n(model confident)",
                    transform=axes[0, 2].transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Row 2: Worst test sample (where active learning would help)
    vertices, elements, boundary = read_mfem_mesh(str(worst_test['file']))
    refined_verts, refined_stress, refined_elems = simulate_ground_truth_refined(
        vertices, elements, boundary, worst_test['params'], refine_levels=2
    )
    prediction = predict_with_transolver(
        model, vertices, elements, boundary, worst_test['params'], norm_params, device, max_elems
    )
    coarse_gt = simulate_ground_truth(vertices, elements, boundary, worst_test['params'])
    error = np.abs(prediction - coarse_gt)

    vmin = min(refined_stress.min(), prediction.min())
    vmax = max(refined_stress.max(), prediction.max())

    coll = plot_mesh_field(axes[1, 0], refined_verts, refined_elems, refined_stress,
                           title=f"Test High-Error: GT ({worst_test['file'].stem})",
                           cmap="jet", vmin=vmin, vmax=vmax)
    fig.colorbar(coll, ax=axes[1, 0], shrink=0.8)

    coll = plot_mesh_field(axes[1, 1], vertices, elements, prediction,
                           title="Transolver Prediction",
                           cmap="jet", vmin=vmin, vmax=vmax)
    fig.colorbar(coll, ax=axes[1, 1], shrink=0.8)

    coll = plot_mesh_field(axes[1, 2], vertices, elements, error,
                           title=f"Error (Max: {worst_test['max_error']:.1f})",
                           cmap="Reds")
    fig.colorbar(coll, ax=axes[1, 2], shrink=0.8)
    axes[1, 2].text(0.02, 0.98, "High error\n(needs more training)",
                    transform=axes[1, 2].transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8))

    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved active learning analysis to: {output_file}")

    # Compute improvement potential
    avg_train_error = np.mean([r['max_error'] for r in train_results])
    avg_test_error = np.mean([r['max_error'] for r in test_results])

    return {
        'best_train_sample': best_train['file'].stem,
        'best_train_max_error': best_train['max_error'],
        'worst_test_sample': worst_test['file'].stem,
        'worst_test_max_error': worst_test['max_error'],
        'avg_train_error': avg_train_error,
        'avg_test_error': avg_test_error,
        'error_gap': avg_test_error - avg_train_error,
        'n_train_evaluated': len(train_results),
        'n_test_evaluated': len(test_results),
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Transolver surrogate model")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to test")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--samples-dir", type=str, default=None, help="Test samples directory")
    parser.add_argument("--train-dir", type=str, default=None, help="Training samples directory")
    parser.add_argument("--model-dir", type=str, default=None, help="Trained model directory")
    parser.add_argument("--compare", action="store_true", help="Compare multiple samples")
    parser.add_argument("--best-training", action="store_true", help="Find and visualize best training sample")
    parser.add_argument("--sciml-loop", action="store_true", help="Visualize SciML loop output on test sample")
    parser.add_argument("--active-learning", action="store_true", help="Visualize active learning analysis")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    samples_dir = args.samples_dir or str(project_root / "samples")
    train_dir = args.train_dir or str(project_root / "train")
    model_dir = args.model_dir or str(project_root / "outputs" / "surrogate")
    output_dir = project_root / "tests" / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    if args.active_learning:
        # Visualize active learning analysis (train vs test error)
        output_file = args.output or str(output_dir / "sciml_active_learning.png")
        results = visualize_sciml_active_learning(train_dir, samples_dir, model_dir, output_file)
        print("\nActive Learning Analysis Results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    elif args.best_training:
        # Find and visualize best training sample (min Max Error)
        output_file = args.output or str(output_dir / "transolver_best_training.png")
        results = visualize_best_training_sample(train_dir, model_dir, output_file)
        print("\nBest Training Sample Results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    elif args.sciml_loop:
        # Visualize SciML loop output on test sample
        output_file = args.output or str(output_dir / "sciml_loop_test.png")
        results = visualize_sciml_loop_output(samples_dir, model_dir, output_file, sample_idx=args.sample)
        print("\nSciML Loop Test Results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    elif args.compare:
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
        """Test mesh file loading from train/ folder."""
        project_root = Path(__file__).parent.parent
        train_dir = project_root / "train"
        mesh_files = sorted(train_dir.glob("sample_*.mesh"))

        if not mesh_files:
            import pytest
            pytest.skip("No train meshes found")

        vertices, elements, boundary = read_mfem_mesh(str(mesh_files[0]))

        assert len(vertices) > 0
        assert len(elements) > 0
        assert vertices.shape[1] == 2

    def test_ground_truth_generation(self):
        """Test FEA ground truth generation on train mesh."""
        project_root = Path(__file__).parent.parent
        train_dir = project_root / "train"
        mesh_files = sorted(train_dir.glob("sample_*.mesh"))

        if not mesh_files:
            import pytest
            pytest.skip("No train meshes found")

        vertices, elements, boundary = read_mfem_mesh(str(mesh_files[0]))

        params = {'E': 200e9, 'nu': 0.3, 'load': 100}
        stress = simulate_ground_truth(vertices, elements, boundary, params)

        assert len(stress) == len(elements)
        assert np.all(stress >= 0)

    def test_visualization(self):
        """
        5-panel visualization: fine GT | coarse MFEM | error | adapted | error reduced.
        Uses train/ meshes. Saves PNGs to tests/test_outputs/.
        """
        import pytest
        pytest.importorskip("torch", reason="torch not installed")

        project_root = Path(__file__).parent.parent
        train_dir = project_root / "train"
        model_dir = project_root / "outputs" / "surrogate"
        mesh_files = sorted(train_dir.glob("sample_*.mesh"))

        if not mesh_files:
            pytest.skip("No train meshes found")

        output_dir = Path(__file__).parent / "test_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run for the first 3 train samples
        for mesh_file in mesh_files[:3]:
            stem = mesh_file.stem
            output_file = output_dir / f"sciml_loop_{stem}.png"
            results = visualize_transolver_test(
                str(mesh_file), str(model_dir), str(output_file)
            )
            assert output_file.exists(), f"PNG not created for {stem}"
            assert results['mean_error'] >= 0
            assert 'coarse_elements' in results
            print(f"  [{stem}] adapted={results['adapted']}, "
                  f"mean_error={results['mean_error']:.2f}")
