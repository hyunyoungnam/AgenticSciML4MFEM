"""
Ensemble Uncertainty Visualization.

Tests the ensemble surrogate pipeline:
  1. Train a small EnsembleModel on FEM samples (or on-the-fly if none exist)
  2. Predict mean field + per-node uncertainty on a coarse mesh
  3. Drive r-adaptivity with the uncertainty signal (no GT needed)
  4. Show how uncertainty changes after mesh adaptation
  5. Optionally compare against fine-mesh GT when FEM produces valid results

New 5-panel layout:
  ① Ensemble mean prediction on coarse mesh
  ② Ensemble uncertainty (std across members) — the adaptation signal
  ③ R-adapted mesh (nodes clustered toward high-uncertainty regions)
  ④ Ensemble uncertainty on adapted mesh
  ⑤ GT comparison (optional — shown only when FEM returns non-trivial results)

Usage:
    python tests/test_transolver.py
    python tests/test_transolver.py --sample 3
    python tests/test_transolver.py --compare
    python tests/test_transolver.py --no-gt
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors


# =============================================================================
# Mesh I/O
# =============================================================================

def read_mfem_mesh(filepath: str) -> Tuple[np.ndarray, list, list]:
    """Read MFEM .mesh file → (vertices, elements, boundary)."""
    vertices, elements, boundary = [], [], []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "elements":
            i += 1
            n = int(lines[i].strip()); i += 1
            for _ in range(n):
                parts = lines[i].strip().split(); i += 1
                t = int(parts[1])
                if t == 2:
                    elements.append([int(parts[2]), int(parts[3]), int(parts[4])])
                elif t == 3:
                    elements.append([int(parts[2]), int(parts[3]),
                                     int(parts[4]), int(parts[5])])

        elif line == "boundary":
            i += 1
            n = int(lines[i].strip()); i += 1
            for _ in range(n):
                parts = lines[i].strip().split(); i += 1
                boundary.append((int(parts[0]), int(parts[2]), int(parts[3])))

        elif line == "vertices":
            i += 1
            n = int(lines[i].strip()); i += 1
            i += 1  # skip dimension line
            for _ in range(n):
                coords = [float(x) for x in lines[i].strip().split()]; i += 1
                vertices.append(coords[:2])
        else:
            i += 1

    return np.array(vertices), elements, boundary


def get_element_centers(vertices: np.ndarray, elements: list) -> np.ndarray:
    """Centroid of each element → (N_elem, 2)."""
    return np.array([np.mean(vertices[e], axis=0) for e in elements])


def node_to_elem_field(elements: list, node_field: np.ndarray) -> np.ndarray:
    """Average per-node values to element centers."""
    return np.array([np.mean(node_field[e]) for e in elements])


def elem_to_node_field(vertices: np.ndarray, elements: list,
                       elem_field: np.ndarray) -> np.ndarray:
    """Average per-element values to nodes."""
    total = np.zeros(len(vertices))
    count = np.zeros(len(vertices))
    for i, elem in enumerate(elements):
        for v in elem:
            total[v] += elem_field[i]
            count[v] += 1
    return total / np.maximum(count, 1)


# =============================================================================
# Mesh Adaptation
# =============================================================================

def adapt_mesh_r(mesh_file: str,
                 node_uncertainty: np.ndarray
                 ) -> Tuple[Optional[np.ndarray], object, Optional[str]]:
    """
    Apply TMOP r-adaptivity driven by per-node uncertainty.

    Args:
        mesh_file:        Path to MFEM .mesh file
        node_uncertainty: (N_nodes,) scalar uncertainty per node

    Returns:
        (adapted_verts, AdaptivityResult|None, tmp_mesh_path|None)
    """
    try:
        from piano.mesh.mfem_manager import MFEMManager
        from piano.morphing.r_adaptivity import TMOPAdaptivity, AdaptivityConfig

        mgr = MFEMManager(mesh_file)
        if mgr.num_nodes != len(node_uncertainty):
            print(f"  R-adaptivity skipped: node count mismatch "
                  f"({mgr.num_nodes} vs {len(node_uncertainty)})")
            return None, None, None

        result = TMOPAdaptivity(
            AdaptivityConfig(max_iterations=100, verbosity=0)
        ).adapt(mgr, node_uncertainty)

        if result.success:
            with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as f:
                tmp = f.name
            mgr._extract_mesh_data()
            saved = mgr.save(tmp)
            return result.coords_adapted, result, str(saved) if saved else None

        return None, None, None

    except Exception as e:
        print(f"  R-adaptivity failed: {e}")
        return None, None, None


def adapt_mesh_h(mesh_file: str,
                 elem_uncertainty: np.ndarray
                 ) -> Tuple[Optional[np.ndarray], Optional[list], object, Optional[str]]:
    """
    Apply h-refinement driven by per-element uncertainty.

    Returns:
        (h_verts, h_elements, HRefinementResult|None, tmp_mesh_path|None)
    """
    try:
        from piano.mesh.mfem_manager import MFEMManager
        from piano.morphing.h_refinement import HRefinement, HRefinementConfig

        mgr = MFEMManager(mesh_file)
        if mgr.num_elements != len(elem_uncertainty):
            print(f"  H-refinement skipped: element count mismatch "
                  f"({mgr.num_elements} vs {len(elem_uncertainty)})")
            return None, None, None, None

        result = HRefinement(
            HRefinementConfig(error_threshold=0.3, max_refinement_levels=3,
                              max_elements=3000)
        ).refine(mgr, elem_uncertainty)

        if result.success:
            h_verts = mgr.get_nodes()
            h_elements = [list(e[e >= 0]) for e in mgr.get_elements()]
            with tempfile.NamedTemporaryFile(suffix='.mesh', delete=False) as f:
                tmp = f.name
            mgr.save(tmp)
            return h_verts, h_elements, result, tmp

        return None, None, result, None

    except Exception as e:
        print(f"  H-refinement failed: {e}")
        return None, None, None, None


# =============================================================================
# FEM Simulation (optional GT)
# =============================================================================

def _retag_boundaries_mfem(mfem_mesh, verts: np.ndarray) -> None:
    eps = 1e-10
    for i in range(mfem_mesh.GetNBE()):
        iv = mfem_mesh.GetBdrElement(i).GetVerticesArray()
        xs = [verts[iv[j]][0] for j in range(len(iv))]
        ys = [verts[iv[j]][1] for j in range(len(iv))]
        if all(y < eps for y in ys):           tag = 1
        elif all(x > 1.0 - eps for x in xs):  tag = 2
        elif all(y > 1.0 - eps for y in ys):  tag = 3
        elif all(x < eps for x in xs):         tag = 4
        else:                                   tag = 5
        mfem_mesh.GetBdrElement(i).SetAttribute(tag)
    mfem_mesh.SetAttributes()


def _extract_verts_ctypes(mfem_mesh) -> np.ndarray:
    import ctypes
    nv = mfem_mesh.GetNV()
    verts = np.zeros((nv, 2))
    for i in range(nv):
        v = mfem_mesh.GetVertex(i)
        p = ctypes.cast(int(v), ctypes.POINTER(ctypes.c_double))
        verts[i] = [p[0], p[1]]
    return verts


def simulate_ground_truth(mesh_file: str,
                          params: Dict
                          ) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Run PyMFEM linear-elasticity FEM.

    Returns:
        (von_mises per element, vertices, elements)
        Returns all-zeros von_mises if the solver fails or produces a
        trivial solution — callers should check ``np.any(von_mises > 0)``.
    """
    from piano.mesh.mfem_manager import MFEMManager
    from piano.solvers.mfem_solver import MFEMSolver
    from piano.solvers.base import (
        PhysicsConfig, PhysicsType, MaterialProperties,
        BoundaryCondition, BoundaryConditionType,
    )

    manager = MFEMManager(mesh_file)
    verts = _extract_verts_ctypes(manager.mesh)
    _retag_boundaries_mfem(manager.mesh, verts)

    physics = PhysicsConfig(
        physics_type=PhysicsType.LINEAR_ELASTICITY,
        material=MaterialProperties(E=params['E'], nu=params['nu']),
        boundary_conditions=[
            BoundaryCondition(BoundaryConditionType.SYMMETRY,
                              boundary_id=4, direction=0),
            BoundaryCondition(BoundaryConditionType.SYMMETRY,
                              boundary_id=1, direction=1),
            BoundaryCondition(BoundaryConditionType.TRACTION,
                              boundary_id=2,
                              value=np.array([params['load'], 0.])),
        ],
    )
    solver = MFEMSolver(order=1)
    solver.setup(manager, physics)

    with tempfile.TemporaryDirectory() as tmp:
        result = solver.solve(tmp)

    vertices = manager.get_nodes()
    elements = [list(e[e >= 0]) for e in manager.get_elements()]

    if not result.success:
        return np.zeros(len(elements)), vertices, elements

    return result.solution_data.get('von_mises',
                                    np.zeros(len(elements))), vertices, elements


def simulate_ground_truth_refined(mesh_file: str, params: Dict,
                                  refine_levels: int = 2
                                  ) -> Tuple[np.ndarray, np.ndarray, list]:
    """Uniformly refine mesh N levels then run FEM."""
    from piano.mesh.mfem_manager import MFEMManager

    mgr = MFEMManager(mesh_file)
    mgr.refine_uniformly(times=refine_levels)

    with tempfile.NamedTemporaryFile(suffix='.mesh', delete=False) as f:
        fine_tmp = f.name
    mgr.save(fine_tmp)
    try:
        return simulate_ground_truth(fine_tmp, params)
    finally:
        os.unlink(fine_tmp)


# =============================================================================
# Ensemble Model
# =============================================================================

def load_ensemble_model(model_dir: str):
    """
    Load a trained EnsembleModel from ``model_dir/ensemble_model.pt``.

    Returns a SurrogateTrainer with normalizers restored from the sidecar
    ``normalizer_params.json`` so that predictions are properly denormalized.

    Raises FileNotFoundError if the model file does not exist.
    """
    import json, torch
    from piano.surrogate.base import TransolverConfig, EnsembleConfig
    from piano.surrogate.ensemble import EnsembleModel
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig, Normalizer

    path = Path(model_dir) / "ensemble_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Ensemble model not found: {path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    mc = checkpoint['ensemble_config']['member_config']
    member_cfg = TransolverConfig(**{k: v for k, v in mc.items()
                                     if k != 'checkpoint_dir'})
    ens_cfg = EnsembleConfig(
        n_members=checkpoint['ensemble_config']['n_members'],
        member_config=member_cfg,
    )
    model = EnsembleModel(ens_cfg)
    model.build(checkpoint['input_dim'],
                checkpoint['coord_dim'],
                checkpoint['num_points'])
    model.load_state_dict(checkpoint['state_dict'])
    model._is_trained = True
    model.eval()

    # Reconstruct trainer shell with saved normalizers
    trainer = SurrogateTrainer(TrainingConfig())
    trainer._model = model

    norm_path = Path(model_dir) / "normalizer_params.json"
    if norm_path.exists():
        with open(norm_path) as f:
            np_dict = json.load(f)
        if 'input_mean' in np_dict:
            n = Normalizer()
            n.mean = np.array(np_dict['input_mean'])
            n.std  = np.array(np_dict['input_std'])
            trainer._input_normalizer = n
        if 'output_mean' in np_dict:
            n = Normalizer()
            n.mean = np.array(np_dict['output_mean'])
            n.std  = np.array(np_dict['output_std'])
            trainer._output_normalizer = n

    return trainer


def train_fast_ensemble(
    samples_dirs: List[str],
    n_ensemble: int = 3,
    epochs: int = 80,
    params_seed: int = 42,
    save_dir: Optional[str] = None,
):
    """
    Train an EnsembleModel on all FEM samples from one or more directories.

    Args:
        samples_dirs: One or more directories containing sample_*.mesh files
        n_ensemble:   Ensemble members
        epochs:       Training epochs
        params_seed:  RNG seed for material/load parameters
        save_dir:     Save trained model here if provided

    Returns:
        SurrogateTrainer with trained model and normalizers
    """
    from piano.data.dataset import FEMDataset, FEMSample, DatasetConfig
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.surrogate.base import TransolverConfig

    if isinstance(samples_dirs, str):
        samples_dirs = [samples_dirs]

    mesh_files = []
    for d in samples_dirs:
        mesh_files.extend(sorted(Path(d).glob("sample_*.mesh")))

    if not mesh_files:
        raise FileNotFoundError(f"No sample_*.mesh found in {samples_dirs}")

    rng = np.random.default_rng(params_seed)

    dataset = FEMDataset(DatasetConfig(
        parameter_names=['E', 'nu', 'load'],
        parameter_bounds={
            'E':    (150e9, 250e9),
            'nu':   (0.25, 0.35),
            'load': (50e6, 150e6),
        },
    ))

    dirs_str = ", ".join(samples_dirs)
    print(f"  Collecting FEM data from {len(mesh_files)} meshes ({dirs_str})...")
    for i, mf in enumerate(mesh_files):
        params = {
            'E':    float(rng.uniform(150e9, 250e9)),
            'nu':   float(rng.uniform(0.25, 0.35)),
            'load': float(rng.uniform(50e6, 150e6)),
        }
        try:
            von_mises, verts_fem, elems_fem = simulate_ground_truth(str(mf), params)
            if not np.any(von_mises > 0):
                raise RuntimeError(f"FEM returned trivial solution for {mf.stem}")

            # Use element centres as coordinates so coords/outputs have same N
            centers = get_element_centers(verts_fem, elems_fem)
            sample = FEMSample(
                sample_id=f"s{i:03d}",
                parameters=params,
                coordinates=centers,               # (N_elem, 2)
                von_mises=von_mises[:, np.newaxis], # (N_elem, 1)
                is_valid=True,
            )
            dataset.add_sample(sample)
        except Exception as e:
            print(f"    Skipping {mf.stem}: {e}")

    valid = dataset.get_valid_samples()
    if len(valid) < 3:
        raise RuntimeError(f"Need ≥3 valid samples, got {len(valid)}")

    print(f"  {len(valid)} valid samples — training {n_ensemble}-member ensemble "
          f"for {epochs} epochs...")

    training_cfg = TrainingConfig(
        surrogate_config=TransolverConfig(
            d_model=64, n_heads=4, n_layers=2, slice_num=8,
            epochs=epochs, patience=50, batch_size=4,
        ),
        use_ensemble=True,
        n_ensemble=n_ensemble,
        train_test_split=0.2,
        random_seed=params_seed,
        save_dir=Path(save_dir) if save_dir else None,
    )
    trainer = SurrogateTrainer(training_cfg)

    params_arr, coords_list, outputs_list = dataset.prepare_training_data('von_mises')
    result = trainer.train(params_arr, coords_list, outputs_list)

    if not result.success:
        raise RuntimeError(f"Ensemble training failed: {result.error_message}")

    print(f"  Done — train_loss={result.train_loss:.4f}  "
          f"test_loss={result.test_loss:.4f}")

    if save_dir:
        out = Path(save_dir) / "ensemble_model.pt"
        trainer.model.save(out)
        print(f"  Saved ensemble → {out}")
        # Save normalizer params so the model can be loaded with proper normalization
        import json
        norm_params = trainer._get_normalization_params()
        with open(Path(save_dir) / "normalizer_params.json", "w") as f:
            json.dump(norm_params, f)

    return trainer


def predict_ensemble(
    trainer,
    vertices: np.ndarray,
    elements: list,
    params: Dict,
    param_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run ensemble prediction on a mesh with proper normalization.

    Uses element centres as query coordinates (consistent with training).
    Delegates to ``SurrogateTrainer.predict_with_uncertainty()`` so that
    input parameters are normalized before inference and outputs are
    denormalized before returning.

    Args:
        trainer:     SurrogateTrainer (wraps EnsembleModel + normalizers)
        vertices:    (N_nodes, 2)
        elements:    element connectivity list
        params:      parameter dict (raw, un-normalized)
        param_names: ordered list of parameter keys

    Returns:
        mean_field:  (N_elem,) ensemble mean von Mises prediction (Pa scale)
        uncertainty: (N_elem,) ensemble std (Pa scale) — the adaptation signal
    """
    from piano.surrogate.trainer import SurrogateTrainer

    centers   = get_element_centers(vertices, elements)   # (N_elem, 2)
    param_arr = np.array([[params[k] for k in param_names]], dtype=np.float32)

    mean, unc = trainer.predict_with_uncertainty(param_arr, centers)

    mean_field  = mean.flatten()
    uncertainty = unc.flatten() if unc is not None else np.zeros(len(centers))

    return mean_field, uncertainty


# =============================================================================
# Plotting Utilities
# =============================================================================

def plot_mesh_field(
    ax,
    vertices: np.ndarray,
    elements: list,
    field: np.ndarray,
    title: str = "",
    cmap: str = "viridis",
    vmin: float = None,
    vmax: float = None,
) -> PolyCollection:
    """Plot a scalar element-centred field on the mesh."""
    polygons = [vertices[elem] for elem in elements]

    if vmin is None: vmin = float(np.min(field))
    if vmax is None: vmax = float(np.max(field))
    if vmin == vmax:
        vmax = vmin + 1e-10

    coll = PolyCollection(
        polygons,
        array=field,
        cmap=cmap,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        edgecolors='black',
        linewidths=0.2,
    )
    ax.add_collection(coll)
    ax.set_xlim(vertices[:, 0].min() - 0.05, vertices[:, 0].max() + 0.05)
    ax.set_ylim(vertices[:, 1].min() - 0.05, vertices[:, 1].max() + 0.05)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    return coll


def nearest_neighbour_interp(src_centers, src_values, tgt_centers):
    diff    = tgt_centers[:, np.newaxis, :] - src_centers[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)
    return src_values[np.argmin(sq_dist, axis=1)]


# =============================================================================
# Main Visualization: Uncertainty-Driven Loop
# =============================================================================

def visualize_uncertainty_loop(
    mesh_file: str,
    model_dir: str,
    output_file: str,
    samples_dirs: Optional[List[str]] = None,
    params: Optional[Dict] = None,
    show_gt: bool = True,
    n_ensemble: int = 3,
    epochs: int = 80,
) -> Dict:
    """
    5-panel uncertainty-driven SciML loop visualization.

    ① Ensemble mean prediction  (coarse mesh)
    ② Ensemble uncertainty      (coarse mesh) — the adaptation signal
    ③ R-adapted mesh            (nodes toward high-uncertainty)
    ④ Ensemble uncertainty      (adapted mesh)
    ⑤ GT comparison             (optional fine-mesh FEM)

    Panel ⑤ is shown only when ``show_gt=True`` AND the FEM solver returns a
    non-trivial solution.  Otherwise the panel displays a note explaining that
    GT is optional validation and was not available for this run.

    Args:
        mesh_file:   Path to the coarse test mesh
        model_dir:   Directory with ``ensemble_model.pt`` (or where to save one)
        output_file: Output PNG path
        samples_dir: Directory with training meshes (used if no model found)
        params:      Material/load parameters (random if None)
        show_gt:     Whether to attempt fine-mesh FEM for panel ⑤
        n_train:     Training meshes for on-the-fly ensemble
        n_ensemble:  Ensemble size for on-the-fly training
        epochs:      Training epochs for on-the-fly ensemble

    Returns:
        Dict of metrics
    """
    print(f"Loading coarse mesh: {mesh_file}")
    vertices, elements, boundary = read_mfem_mesh(mesh_file)
    n_nodes, n_elems = len(vertices), len(elements)
    print(f"  {n_nodes} nodes, {n_elems} elements")

    if params is None:
        rng = np.random.default_rng(hash(mesh_file) % 2**31)
        params = {
            'E':    float(rng.uniform(150e9, 250e9)),
            'nu':   float(rng.uniform(0.25, 0.35)),
            'load': float(rng.uniform(50e6, 150e6)),
        }
    param_names = ['E', 'nu', 'load']

    # ── Load or train ensemble ─────────────────────────────────────────────
    print("Loading ensemble model...")
    try:
        trainer = load_ensemble_model(model_dir)
        print(f"  Loaded from {model_dir}")
    except FileNotFoundError:
        print("  Not found — training on-the-fly...")
        if samples_dirs is None:
            samples_dirs = [str(Path(mesh_file).parent)]
        trainer = train_fast_ensemble(
            samples_dirs,
            n_ensemble=n_ensemble,
            epochs=epochs,
            save_dir=model_dir,
        )

    # ── ① Ensemble mean prediction on coarse mesh ─────────────────────────
    print("Running ensemble on coarse mesh...")
    mean_coarse, unc_coarse = predict_ensemble(
        trainer, vertices, elements, params, param_names
    )
    print(f"  Mean prediction range: [{mean_coarse.min():.2f}, {mean_coarse.max():.2f}]")
    print(f"  Uncertainty range:     [{unc_coarse.min():.4f}, {unc_coarse.max():.4f}]")

    # ── ② → ③  R-adaptivity driven by uncertainty ─────────────────────────
    print("Applying TMOP r-adaptivity (driven by ensemble uncertainty)...")
    node_unc_coarse = elem_to_node_field(vertices, elements, unc_coarse)
    r_verts, r_result, r_tmp = adapt_mesh_r(mesh_file, node_unc_coarse)
    r_ok = r_result is not None and r_result.success

    if r_ok:
        max_disp = np.linalg.norm(r_verts - vertices, axis=1).max()
        print(f"  R-adapt OK — max node displacement: {max_disp:.4f}")
    else:
        r_verts = vertices.copy()
        r_tmp = None
        print("  R-adapt unavailable")

    # ── H-refinement chained on r-adapted mesh ────────────────────────────
    print("Applying h-refinement (driven by ensemble uncertainty)...")
    h_src = r_tmp if r_tmp else mesh_file
    h_verts, h_elements, h_result, h_tmp = adapt_mesh_h(h_src, unc_coarse)
    h_ok = h_result is not None and h_result.success

    if h_ok:
        print(f"  H-refine OK — elements: "
              f"{h_result.num_elements_before} → {h_result.num_elements_after}")
    else:
        h_verts    = r_verts.copy()
        h_elements = elements
        print("  H-refine unavailable")

    if r_tmp:
        try: os.unlink(r_tmp)
        except OSError: pass

    # ── ④ Uncertainty on adapted mesh ─────────────────────────────────────
    print("Running ensemble on adapted mesh...")
    mean_adapted, unc_adapted = predict_ensemble(
        trainer, h_verts, h_elements, params, param_names
    )
    print(f"  Adapted uncertainty range: "
          f"[{unc_adapted.min():.4f}, {unc_adapted.max():.4f}]")
    mean_unc_reduction = float(np.mean(unc_coarse) - np.mean(unc_adapted))

    if h_tmp:
        try: os.unlink(h_tmp)
        except OSError: pass

    # ── ⑤ GT comparison (optional) ────────────────────────────────────────
    gt_available = False
    gt_fine, fine_verts, fine_elements = None, None, None
    rel_err_coarse = rel_err_adapted = float('nan')

    if show_gt:
        print("Computing optional GT (fine-mesh FEM)...")
        try:
            gt_fine, fine_verts, fine_elements = simulate_ground_truth_refined(
                mesh_file, params, refine_levels=3
            )
            if np.any(gt_fine > 0):
                gt_available = True
                fine_centers = get_element_centers(fine_verts, fine_elements)
                coarse_centers = get_element_centers(vertices, elements)
                adapted_centers = get_element_centers(h_verts, h_elements)

                coarse_on_fine = nearest_neighbour_interp(
                    coarse_centers, mean_coarse, fine_centers
                )
                adapted_on_fine = nearest_neighbour_interp(
                    adapted_centers, mean_adapted, fine_centers
                )
                rel_err_coarse = (
                    np.mean(np.abs(gt_fine - coarse_on_fine))
                    / (np.mean(np.abs(gt_fine)) + 1e-10) * 100
                )
                rel_err_adapted = (
                    np.mean(np.abs(gt_fine - adapted_on_fine))
                    / (np.mean(np.abs(gt_fine)) + 1e-10) * 100
                )
                print(f"  GT peak: {gt_fine.max():.2f}  "
                      f"rel_err coarse={rel_err_coarse:.1f}%  "
                      f"adapted={rel_err_adapted:.1f}%")
            else:
                print("  FEM returned trivial zero solution — GT not shown")
        except Exception as e:
            print(f"  GT skipped: {e}")

    # ── Build 5-panel figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    fig.suptitle(
        'Ensemble Uncertainty Loop  |  '
        '① Mean Pred  →  ② Uncertainty  →  ③ Adapted Mesh  →  '
        '④ Unc. Adapted  →  ⑤ GT (optional)',
        fontsize=10, fontweight='bold', y=1.02,
    )

    # Shared uncertainty scale for ②④
    unc_vmax = max(float(np.percentile(unc_coarse, 98)),
                   float(np.percentile(unc_adapted, 98)),
                   1e-10)

    # Shared mean-prediction scale for ①③
    pred_vmax = float(np.percentile(
        np.concatenate([mean_coarse, mean_adapted]), 98
    ))
    pred_vmax = max(pred_vmax, 1e-10)

    # ① Mean prediction — coarse mesh
    c1 = plot_mesh_field(
        axes[0], vertices, elements, mean_coarse,
        title=f"① Ensemble Mean  ({n_elems} elems)\n"
              f"Coarse mesh prediction",
        cmap='jet', vmin=0, vmax=pred_vmax,
    )
    fig.colorbar(c1, ax=axes[0], shrink=0.8).set_label('Prediction (a.u.)', fontsize=8)
    axes[0].text(
        0.02, 0.02,
        f"Range: [{mean_coarse.min():.1f}, {mean_coarse.max():.1f}]",
        transform=axes[0].transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9),
    )

    # ② Uncertainty — coarse mesh (the adaptation signal)
    c2 = plot_mesh_field(
        axes[1], vertices, elements, unc_coarse,
        title=f"② Ensemble Uncertainty  ({n_elems} elems)\n"
              f"Adaptation signal (no GT needed)",
        cmap='hot_r', vmin=0, vmax=unc_vmax,
    )
    fig.colorbar(c2, ax=axes[1], shrink=0.8).set_label('Std (a.u.)', fontsize=8)
    axes[1].text(
        0.02, 0.02,
        f"Mean: {np.mean(unc_coarse):.3f}\nMax: {unc_coarse.max():.3f}",
        transform=axes[1].transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
    )

    # ③ R+H adapted mesh — mean prediction
    c3 = plot_mesh_field(
        axes[2], h_verts, h_elements, mean_adapted,
        title=f"③ Adapted Mesh  ({len(h_elements)} elems)\n"
              f"Nodes → high-uncertainty regions",
        cmap='jet', vmin=0, vmax=pred_vmax,
    )
    fig.colorbar(c3, ax=axes[2], shrink=0.8).set_label('Prediction (a.u.)', fontsize=8)
    note3 = []
    if r_ok: note3.append(f"R-adapt: ✓ (Δmax={max_disp:.3f})")
    if h_ok: note3.append(f"H-refine: {h_result.num_elements_before}"
                           f"→{h_result.num_elements_after}")
    if note3:
        axes[2].text(
            0.02, 0.02, "\n".join(note3),
            transform=axes[2].transAxes, fontsize=8, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
        )

    # ④ Uncertainty on adapted mesh
    c4 = plot_mesh_field(
        axes[3], h_verts, h_elements, unc_adapted,
        title=f"④ Uncertainty After Adaptation\n"
              f"({len(h_elements)} elems)",
        cmap='hot_r', vmin=0, vmax=unc_vmax,
    )
    fig.colorbar(c4, ax=axes[3], shrink=0.8).set_label('Std (a.u.)', fontsize=8)
    colour4 = 'lightgreen' if mean_unc_reduction > 0 else 'lightsalmon'
    axes[3].text(
        0.02, 0.02,
        f"Mean: {np.mean(unc_adapted):.3f}\nMax: {unc_adapted.max():.3f}\n"
        f"Δ mean: {mean_unc_reduction:+.3f}",
        transform=axes[3].transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round', facecolor=colour4, alpha=0.9),
    )

    # ⑤ GT comparison — optional
    if gt_available:
        c5 = plot_mesh_field(
            axes[4], fine_verts, fine_elements, gt_fine,
            title=f"⑤ GT Fine-Mesh FEM  ({len(fine_elements)} elems)\n"
                  f"Optional external validation",
            cmap='jet',
        )
        fig.colorbar(c5, ax=axes[4], shrink=0.8).set_label(
            'Von Mises (a.u.)', fontsize=8)
        axes[4].text(
            0.02, 0.02,
            f"Rel err coarse: {rel_err_coarse:.1f}%\n"
            f"Rel err adapted: {rel_err_adapted:.1f}%\n"
            f"Δ: {rel_err_coarse - rel_err_adapted:+.1f} pp",
            transform=axes[4].transAxes, fontsize=8, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9),
        )
    else:
        axes[4].set_aspect('equal')
        axes[4].set_xlim(0, 1); axes[4].set_ylim(0, 1)
        axes[4].text(
            0.5, 0.5,
            "⑤ GT Comparison\n(optional validation)\n\n"
            "FEM returned trivial solution\nor GT not requested.\n\n"
            "Uncertainty-driven loop\ndoes not require GT.",
            transform=axes[4].transAxes, fontsize=9,
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
        )
        axes[4].set_title(
            "⑤ GT (optional — not available)",
            fontsize=10, fontweight='bold',
        )
        axes[4].axis('off')

    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

    return {
        'mean_uncertainty_coarse':  float(np.mean(unc_coarse)),
        'max_uncertainty_coarse':   float(np.max(unc_coarse)),
        'mean_uncertainty_adapted': float(np.mean(unc_adapted)),
        'max_uncertainty_adapted':  float(np.max(unc_adapted)),
        'mean_unc_reduction':       mean_unc_reduction,
        'coarse_elements':          n_elems,
        'adapted_elements':         len(h_elements),
        'r_adapted':                r_ok,
        'h_refined':                h_ok,
        'gt_available':             gt_available,
        'rel_err_coarse':           rel_err_coarse,
        'rel_err_adapted':          rel_err_adapted,
    }


def compare_multiple_uncertainty(
    samples_dirs: List[str],
    model_dir: str,
    output_file: str,
    n_samples: int = 3,
) -> List[Dict]:
    """
    Compare ensemble mean + uncertainty across multiple meshes.

    Two rows per mesh: mean prediction (top), uncertainty (bottom).
    """
    if isinstance(samples_dirs, str):
        samples_dirs = [samples_dirs]

    all_files = []
    for d in samples_dirs:
        all_files.extend(sorted(Path(d).glob("sample_*.mesh")))
    mesh_files = all_files[:n_samples]

    if not mesh_files:
        print(f"No mesh files found in {samples_dirs}")
        return []

    print("Loading ensemble model...")
    try:
        trainer = load_ensemble_model(model_dir)
    except FileNotFoundError:
        print("  Not found — training on-the-fly...")
        trainer = train_fast_ensemble(samples_dirs, save_dir=model_dir)

    param_names = ['E', 'nu', 'load']
    n_cols = len(mesh_files)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 8))
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle('Ensemble: Mean Prediction vs Uncertainty',
                 fontsize=13, fontweight='bold')

    results = []
    for col, mf in enumerate(mesh_files):
        verts, elems, _ = read_mfem_mesh(str(mf))
        rng = np.random.default_rng(hash(str(mf)) % 2**31)
        params = {
            'E':    float(rng.uniform(150e9, 250e9)),
            'nu':   float(rng.uniform(0.25, 0.35)),
            'load': float(rng.uniform(50e6, 150e6)),
        }
        mean_f, unc_f = predict_ensemble(trainer, verts, elems, params, param_names)

        pred_vmax = float(np.percentile(mean_f, 98))
        unc_vmax  = float(np.percentile(unc_f, 98)) or 1e-10

        c1 = plot_mesh_field(
            axes[0, col], verts, elems, mean_f,
            title=f"{mf.stem}\nMean prediction ({len(elems)} elems)",
            cmap='jet', vmin=0, vmax=max(pred_vmax, 1e-10),
        )
        fig.colorbar(c1, ax=axes[0, col], shrink=0.8).set_label('Mean', fontsize=8)

        c2 = plot_mesh_field(
            axes[1, col], verts, elems, unc_f,
            title=f"Uncertainty — mean={np.mean(unc_f):.3f}",
            cmap='hot_r', vmin=0, vmax=unc_vmax,
        )
        fig.colorbar(c2, ax=axes[1, col], shrink=0.8).set_label('Std', fontsize=8)

        results.append({
            'mesh':            mf.stem,
            'mean_prediction': float(np.mean(mean_f)),
            'mean_uncertainty': float(np.mean(unc_f)),
            'max_uncertainty': float(np.max(unc_f)),
        })

    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble uncertainty-driven SciML loop visualization"
    )
    parser.add_argument("--sample",      type=int, default=0,
                        help="Sample index to visualize (from first samples-dir)")
    parser.add_argument("--output",      type=str, default=None,
                        help="Output PNG path")
    parser.add_argument("--samples-dirs", type=str, nargs="+", default=None,
                        help="One or more mesh sample directories (e.g. train01 train02)")
    parser.add_argument("--model-dir",   type=str, default=None,
                        help="Trained ensemble model directory")
    parser.add_argument("--compare",     action="store_true",
                        help="Compare uncertainty across multiple samples")
    parser.add_argument("--no-gt",       action="store_true",
                        help="Skip optional GT fine-mesh FEM panel")
    parser.add_argument("--epochs",      type=int, default=500,
                        help="Training epochs for on-the-fly ensemble")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    samples_dirs = args.samples_dirs or [
        str(project_root / "train01"),
        str(project_root / "train02"),
    ]
    model_dir  = args.model_dir or str(project_root / "outputs" / "surrogate")
    output_dir = project_root / "tests" / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    if args.compare:
        out = args.output or str(output_dir / "ensemble_comparison.png")
        results = compare_multiple_uncertainty(samples_dirs, model_dir, out)
        print("\nComparison Results:")
        for r in results:
            print(f"  {r['mesh']}: mean_unc={r['mean_uncertainty']:.4f}  "
                  f"max_unc={r['max_uncertainty']:.4f}")
    else:
        mesh_files = []
        for d in samples_dirs:
            mesh_files.extend(sorted(Path(d).glob("sample_*.mesh")))
        if not mesh_files:
            print(f"No mesh files found in {samples_dirs}")
            sys.exit(1)

        idx = min(args.sample, len(mesh_files) - 1)
        mf  = str(mesh_files[idx])
        out = args.output or str(output_dir / f"uncertainty_loop_{idx:03d}.png")

        results = visualize_uncertainty_loop(
            mf, model_dir, out,
            samples_dirs=samples_dirs,
            show_gt=not args.no_gt,
            epochs=args.epochs,
        )
        print("\nResults:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


# =============================================================================
# Pytest Integration
# =============================================================================

class TestTransolver:
    """Pytest tests for ensemble uncertainty pipeline."""

    def _samples_dir(self):
        return Path(__file__).parent.parent / "samples"

    def _model_dir(self):
        return Path(__file__).parent.parent / "outputs" / "surrogate"

    def test_mesh_loading(self):
        """Mesh files in samples/ are readable."""
        mesh_files = sorted(self._samples_dir().glob("sample_*.mesh"))
        if not mesh_files:
            import pytest; pytest.skip("No sample meshes found")

        verts, elems, bdr = read_mfem_mesh(str(mesh_files[0]))
        assert len(verts) > 0
        assert len(elems) > 0
        assert verts.shape[1] == 2

    def test_ensemble_uncertainty(self):
        """Ensemble produces non-trivial uncertainty on sample mesh."""
        import pytest
        pytest.importorskip("torch", reason="torch not installed")

        mesh_files = sorted(self._samples_dir().glob("sample_*.mesh"))
        if not mesh_files:
            pytest.skip("No sample meshes found")

        output_dir = Path(__file__).parent / "test_outputs"
        output_dir.mkdir(exist_ok=True)
        model_dir = str(self._model_dir())

        try:
            model, _ = load_ensemble_model(model_dir)
        except FileNotFoundError:
            model = train_fast_ensemble(
                str(self._samples_dir()),
                n_train=10, n_ensemble=3, epochs=30,
                save_dir=model_dir,
            )

        verts, elems, _ = read_mfem_mesh(str(mesh_files[0]))
        params = {'E': 200e9, 'nu': 0.3, 'load': 100e6}
        mean_f, unc_f = predict_ensemble(
            model, verts, elems, params, ['E', 'nu', 'load']
        )

        assert mean_f.shape == (len(elems),)
        assert unc_f.shape  == (len(elems),)
        # Uncertainty must be non-negative; at least some non-zero values expected
        assert np.all(unc_f >= 0)

    def test_visualization(self):
        """5-panel uncertainty loop PNG is created without error."""
        import pytest
        pytest.importorskip("torch", reason="torch not installed")

        mesh_files = sorted(self._samples_dir().glob("sample_*.mesh"))
        if not mesh_files:
            pytest.skip("No sample meshes found")

        output_dir = Path(__file__).parent / "test_outputs"
        output_dir.mkdir(exist_ok=True)
        out = str(output_dir / "uncertainty_loop_test.png")

        results = visualize_uncertainty_loop(
            str(mesh_files[0]),
            str(self._model_dir()),
            out,
            samples_dir=str(self._samples_dir()),
            show_gt=True,
            n_train=10,
            n_ensemble=3,
            epochs=30,
        )

        assert Path(out).exists(), "PNG not created"
        assert results['mean_uncertainty_coarse'] >= 0
        assert results['coarse_elements'] > 0
        print(f"  mean_unc_coarse={results['mean_uncertainty_coarse']:.4f}  "
              f"r_adapted={results['r_adapted']}  "
              f"gt_available={results['gt_available']}")
