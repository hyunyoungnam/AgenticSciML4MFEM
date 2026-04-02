"""
Transolver Training Script.

Trains a Transolver surrogate model on plate-with-hole FEA data.

Workflow:
1. Load training meshes from train01/ or train02/ folder
2. Run PyMFEM linear-elasticity FEM to generate ground-truth von Mises stress
3. Train Transolver on (mesh coordinates, parameters) -> stress field mapping
4. Save trained model to outputs/surrogate/

Usage:
    python train_transolver.py
    python train_transolver.py --train-dir train02 --epochs 500 --batch-size 16

Boundary conditions (uniaxial tension):
    Left   (tag 4): u_x = 0  (SYMMETRY, direction=0) — prevents x-translation
    Bottom (tag 1): u_y = 0  (SYMMETRY, direction=1) — prevents y-translation/rotation
    Right  (tag 2): traction [σ₀, 0]  — applied uniaxial load
    Top / hole: traction-free (natural Neumann, no action needed)
"""

import ctypes
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import torch

from meshforge.mesh.mfem_manager import MFEMManager
from meshforge.solvers.mfem_solver import MFEMSolver
from meshforge.solvers.base import (
    BoundaryCondition,
    BoundaryConditionType,
    MaterialProperties,
    PhysicsConfig,
    PhysicsType,
)


# =============================================================================
# Mesh Utilities
# =============================================================================

def get_element_centers(manager: MFEMManager) -> np.ndarray:
    """Compute element centroids from MFEMManager."""
    nodes = manager.get_nodes()       # (N_nodes, 2)
    elements = manager.get_elements() # (N_elems, max_nodes), -1 padded
    centers = []
    for row in elements:
        valid = row[row >= 0]
        centers.append(nodes[valid].mean(axis=0))
    return np.array(centers)


def retag_boundaries(mfem_mesh, verts: np.ndarray) -> None:
    """
    Assign boundary tags geometrically for a unit-square domain [0,1]x[0,1].

    Tags:
        1 = bottom  (y ≈ 0)
        2 = right   (x ≈ 1)
        3 = top     (y ≈ 1)
        4 = left    (x ≈ 0)
        5 = hole    (interior boundary)
    """
    eps = 1e-10
    for i in range(mfem_mesh.GetNBE()):
        iv = mfem_mesh.GetBdrElement(i).GetVerticesArray()
        xs = [verts[iv[j]][0] for j in range(len(iv))]
        ys = [verts[iv[j]][1] for j in range(len(iv))]
        if all(y < eps for y in ys):
            tag = 1  # bottom
        elif all(x > 1.0 - eps for x in xs):
            tag = 2  # right
        elif all(y > 1.0 - eps for y in ys):
            tag = 3  # top
        elif all(x < eps for x in xs):
            tag = 4  # left
        else:
            tag = 5  # hole
        mfem_mesh.GetBdrElement(i).SetAttribute(tag)
    mfem_mesh.SetAttributes()


def _extract_verts(mfem_mesh) -> np.ndarray:
    """Extract vertex coordinates from an mfem.Mesh object."""
    nv = mfem_mesh.GetNV()
    verts = np.zeros((nv, 2))
    for i in range(nv):
        v = mfem_mesh.GetVertex(i)
        p = ctypes.cast(int(v), ctypes.POINTER(ctypes.c_double))
        verts[i] = [p[0], p[1]]
    return verts


# =============================================================================
# FEM Simulation
# =============================================================================

def run_fem_simulation(
    mesh_file: str,
    params: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run PyMFEM linear-elasticity solve on a mesh file.

    Args:
        mesh_file: Path to MFEM .mesh file
        params: Dict with keys 'E' (Pa), 'nu', 'load' (traction magnitude)

    Returns:
        von_mises:      Per-element von Mises stress  (N_elements,)
        elem_centers:   Element centroid coordinates  (N_elements, 2)

    Raises:
        RuntimeError if the FEM solve fails
    """
    manager = MFEMManager(mesh_file)

    # Re-tag boundaries geometrically — robust against any mesh generation quirks
    verts = _extract_verts(manager.mesh)
    retag_boundaries(manager.mesh, verts)

    sigma_0 = params['load']

    physics = PhysicsConfig(
        physics_type=PhysicsType.LINEAR_ELASTICITY,
        material=MaterialProperties(E=params['E'], nu=params['nu']),
        boundary_conditions=[
            # Left edge: u_x = 0 (roller — prevents x-translation, allows y-sliding)
            BoundaryCondition(
                bc_type=BoundaryConditionType.SYMMETRY,
                boundary_id=4,
                direction=0,
            ),
            # Bottom edge: u_y = 0 (prevents y-translation / rigid-body rotation)
            BoundaryCondition(
                bc_type=BoundaryConditionType.SYMMETRY,
                boundary_id=1,
                direction=1,
            ),
            # Right edge: applied uniaxial traction [σ₀, 0]
            BoundaryCondition(
                bc_type=BoundaryConditionType.TRACTION,
                boundary_id=2,
                value=np.array([sigma_0, 0.0]),
            ),
        ],
    )

    solver = MFEMSolver(order=1)
    solver.setup(manager, physics)

    with tempfile.TemporaryDirectory() as tmp:
        result = solver.solve(tmp)

    if not result.success:
        raise RuntimeError(f"FEM solve failed for {mesh_file}: {result.error_message}")

    von_mises = result.solution_data['von_mises']   # (N_elements,)
    elem_centers = get_element_centers(manager)      # (N_elements, 2)

    return von_mises, elem_centers


# =============================================================================
# Dataset Preparation
# =============================================================================

def load_training_data(
    train_dirs: Union[str, List[str]],
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load training data by running FEM on each mesh file.

    Args:
        train_dirs: One or more directories containing sample_*.mesh files.
                    All directories are merged into a single dataset.
        max_samples: Cap on total number of samples to load (applied after merging)

    Returns:
        parameters:    (N, 3)               — [E/GPa, nu, load]
        coordinates:   (N, max_elems, 2)    — element centers, zero-padded
        stress_fields: (N, max_elems)        — von Mises stress, zero-padded
        mesh_files:    list of mesh file paths
    """
    if isinstance(train_dirs, str):
        train_dirs = [train_dirs]

    mesh_files = []
    for d in train_dirs:
        files = sorted(Path(d).glob("sample_*.mesh"))
        mesh_files.extend(files)
        print(f"  Found {len(files)} meshes in {d}")

    if max_samples:
        mesh_files = mesh_files[:max_samples]

    print(f"Loading {len(mesh_files)} training samples total via PyMFEM...")

    # First pass: find max element count for padding
    max_elems = 0
    for mesh_file in mesh_files:
        m = MFEMManager(str(mesh_file))
        max_elems = max(max_elems, m.num_elements)
    print(f"Max elements across samples: {max_elems}")

    all_params = []
    all_coords = []
    all_stress = []

    for i, mesh_file in enumerate(mesh_files):
        np.random.seed(1000 + i)
        E    = np.random.uniform(150e9, 250e9)
        nu   = np.random.uniform(0.25, 0.35)
        load = np.random.uniform(80e6, 120e6)

        von_mises, elem_centers = run_fem_simulation(
            str(mesh_file), {'E': E, 'nu': nu, 'load': load}
        )

        n_elems = len(von_mises)

        padded_coords = np.zeros((max_elems, 2), dtype=np.float32)
        padded_coords[:n_elems] = elem_centers

        padded_stress = np.zeros(max_elems, dtype=np.float32)
        padded_stress[:n_elems] = von_mises

        # Parameter vector: normalise E to GPa for numerical stability
        all_params.append(np.array([E / 1e9, nu, load], dtype=np.float32))
        all_coords.append(padded_coords)
        all_stress.append(padded_stress)

        if (i + 1) % 20 == 0:
            print(f"  Solved {i + 1}/{len(mesh_files)} samples")

    parameters    = np.array(all_params)
    coordinates   = np.array(all_coords)
    stress_fields = np.array(all_stress)

    print(f"Data shapes: params={parameters.shape}, "
          f"coords={coordinates.shape}, stress={stress_fields.shape}")

    return parameters, coordinates, stress_fields, [str(f) for f in mesh_files]


# =============================================================================
# Training
# =============================================================================

def train_transolver(
    parameters: np.ndarray,
    coordinates: np.ndarray,
    stress_fields: np.ndarray,
    config: Dict,
    output_dir: str,
) -> Dict:
    """
    Train Transolver model.

    Args:
        parameters:    (N, n_params)
        coordinates:   (N, max_elems, 2)
        stress_fields: (N, max_elems)
        config:        training hyper-parameters
        output_dir:    output directory for model artifacts

    Returns:
        dict with model_path, best_val_loss, epochs_trained
    """
    from meshforge.surrogate.base import TransolverConfig
    from meshforge.surrogate.transolver import TransolverModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    n_samples = len(parameters)
    n_train = int(n_samples * 0.9)

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    # Normalize
    param_mean  = parameters.mean(axis=0)
    param_std   = parameters.std(axis=0) + 1e-8
    stress_mean = stress_fields.mean()
    stress_std  = stress_fields.std() + 1e-8

    params_norm = (parameters    - param_mean)  / param_std
    stress_norm = (stress_fields - stress_mean) / stress_std

    print(f"Stress stats: mean={stress_mean:.4f}, std={stress_std:.4f}")

    train_params = torch.tensor(params_norm[train_idx],  dtype=torch.float32, device=device)
    train_coords = torch.tensor(coordinates[train_idx],  dtype=torch.float32, device=device)
    train_stress = torch.tensor(stress_norm[train_idx],  dtype=torch.float32, device=device).unsqueeze(-1)

    val_params = torch.tensor(params_norm[val_idx], dtype=torch.float32, device=device)
    val_coords = torch.tensor(coordinates[val_idx], dtype=torch.float32, device=device)
    val_stress = torch.tensor(stress_norm[val_idx], dtype=torch.float32, device=device).unsqueeze(-1)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    transolver_config = TransolverConfig(
        slice_num=config.get('slice_num', 32),
        n_heads=config.get('n_heads', 8),
        d_model=config.get('d_model', 128),
        n_layers=config.get('n_layers', 4),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config.get('dropout', 0.1),
        learning_rate=config.get('learning_rate', 1e-3),
        batch_size=config.get('batch_size', 16),
        epochs=config.get('epochs', 200),
        patience=config.get('patience', 50),
        output_dim=1,
    )

    model = TransolverModel(transolver_config)
    model.build(
        input_dim=parameters.shape[1],
        coord_dim=2,
        num_points=coordinates.shape[1],
    )
    model.to(device)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=transolver_config.learning_rate)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion  = torch.nn.MSELoss()
    batch_size = transolver_config.batch_size

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss    = float('inf')
    patience_counter = 0
    best_state       = None

    for epoch in range(transolver_config.epochs):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(len(train_params))

        for i in range(0, len(train_params), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            pred = model.forward(train_params[idx], train_coords[idx])
            loss = criterion(pred, train_stress[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        train_loss = epoch_loss / len(train_params)
        history['train_loss'].append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model.forward(val_params, val_coords), val_stress).item()
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:4d}: train={train_loss:.6f}  val={val_loss:.6f}")

        if patience_counter >= transolver_config.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    model._is_trained = True

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "transolver_model.pt"
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss':   [float(x) for x in history['val_loss']],
            'best_val_loss': float(best_val_loss),
            'config': config,
        }, f, indent=2)

    norm_path = output_path / "normalization.json"
    with open(norm_path, 'w') as f:
        json.dump({
            'param_mean':  param_mean.tolist(),
            'param_std':   param_std.tolist(),
            'stress_mean': float(stress_mean),
            'stress_std':  float(stress_std),
        }, f, indent=2)

    return {
        'model_path':     str(model_path),
        'best_val_loss':  best_val_loss,
        'epochs_trained': len(history['train_loss']),
        'history':        history,
        'model':          model,
        'norm': {
            'param_mean':  param_mean,
            'param_std':   param_std,
            'stress_mean': stress_mean,
            'stress_std':  stress_std,
        },
        'val_idx':     val_idx,
        'val_params':  val_params,
        'val_coords':  val_coords,
        'val_stress':  val_stress,
    }


# =============================================================================
# Visualization
# =============================================================================

def visualize_results(
    results: Dict,
    parameters: np.ndarray,
    coordinates: np.ndarray,
    stress_fields: np.ndarray,
    output_dir: str,
    n_samples: int = 3,
) -> None:
    """
    Generate and save PNG visualizations:
      1. Training / validation loss curve
      2. Predicted vs ground-truth von Mises stress scatter + spatial plot
         for up to n_samples validation samples
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history    = results['history']
    model      = results['model']
    norm       = results['norm']
    val_idx    = results['val_idx']
    val_params = results['val_params']
    val_coords = results['val_coords']
    val_stress = results['val_stress']

    # ------------------------------------------------------------------
    # 1. Loss curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.semilogy(epochs, history['train_loss'], label='Train loss')
    ax.semilogy(epochs, history['val_loss'],   label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss (normalised)')
    ax.set_title('Transolver training history')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.4)
    fig.tight_layout()
    loss_path = output_path / 'training_loss.png'
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {loss_path}")

    # ------------------------------------------------------------------
    # 2. Per-sample prediction vs ground truth
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        pred_norm = model.forward(val_params, val_coords).cpu().numpy()  # (V, N, 1)

    stress_mean = norm['stress_mean']
    stress_std  = norm['stress_std']

    pred_denorm = pred_norm[..., 0] * stress_std + stress_mean   # (V, N)
    true_denorm = val_stress.cpu().numpy()[..., 0] * stress_std + stress_mean

    n_show = min(n_samples, len(val_idx))

    for k in range(n_show):
        orig_idx = val_idx[k]
        coords_k = coordinates[orig_idx]          # (N, 2)
        true_k   = stress_fields[orig_idx]        # (N,)  — original, unpadded
        pred_k   = pred_denorm[k]                 # (N,)

        # Mask out zero-padded elements
        mask = true_k > 0

        x, y   = coords_k[mask, 0], coords_k[mask, 1]
        t_vals = true_k[mask]
        p_vals = pred_k[mask]

        vmin = min(t_vals.min(), p_vals.min())
        vmax = max(t_vals.max(), p_vals.max())

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for ax, vals, title in zip(
            axes[:2],
            [t_vals, p_vals],
            ['FEM ground truth', 'Transolver prediction'],
        ):
            try:
                triang = mtri.Triangulation(x, y)
                tcf = ax.tricontourf(triang, vals, levels=20, cmap='jet',
                                     vmin=vmin, vmax=vmax)
            except Exception:
                sc = ax.scatter(x, y, c=vals, cmap='jet', s=8,
                                vmin=vmin, vmax=vmax)
                tcf = sc
            plt.colorbar(tcf, ax=ax, label='von Mises (Pa)')
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        # Scatter: predicted vs true
        ax = axes[2]
        ax.scatter(t_vals, p_vals, s=6, alpha=0.5)
        lims = [vmin, vmax]
        ax.plot(lims, lims, 'r--', lw=1, label='y = x')
        ax.set_xlabel('FEM stress (Pa)')
        ax.set_ylabel('Predicted stress (Pa)')
        ax.set_title(f'Scatter — sample {orig_idx}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f'Sample {orig_idx}  |  '
            f'E={parameters[orig_idx,0]*1e9/1e9:.0f} GPa  '
            f'ν={parameters[orig_idx,1]:.3f}  '
            f'σ₀={parameters[orig_idx,2]:.1f} Pa',
            fontsize=10,
        )
        fig.tight_layout()
        out = output_path / f'prediction_sample_{orig_idx:03d}.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Transolver surrogate model")
    parser.add_argument("--train-dir",    type=str,   nargs="+", default=["train01"], help="Training data directories (space-separated; all are merged)")
    parser.add_argument("--output-dir",   type=str,   default="outputs/surrogate",    help="Output directory")
    parser.add_argument("--max-samples",  type=int,   default=None,                   help="Max training samples")
    parser.add_argument("--epochs",       type=int,   default=200,                    help="Training epochs")
    parser.add_argument("--batch-size",   type=int,   default=16,                     help="Batch size")
    parser.add_argument("--d-model",      type=int,   default=128,                    help="Hidden dimension")
    parser.add_argument("--n-layers",     type=int,   default=4,                      help="Number of layers")
    parser.add_argument("--slice-num",    type=int,   default=32,                     help="Number of physics slices")
    parser.add_argument("--learning-rate",type=float, default=1e-3,                   help="Learning rate")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    train_dirs = [str(project_root / d) for d in args.train_dir]
    output_dir = project_root / args.output_dir

    parameters, coordinates, stress_fields, _ = load_training_data(
        train_dirs, max_samples=args.max_samples
    )

    config = {
        'epochs':        args.epochs,
        'batch_size':    args.batch_size,
        'd_model':       args.d_model,
        'n_layers':      args.n_layers,
        'slice_num':     args.slice_num,
        'learning_rate': args.learning_rate,
        'n_heads':       8,
        'mlp_ratio':     4.0,
        'dropout':       0.1,
        'patience':      50,
    }

    print("\nTraining configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    results = train_transolver(parameters, coordinates, stress_fields, config, str(output_dir))

    print("\nTraining complete!")
    print(f"  Model saved to:       {results['model_path']}")
    print(f"  Best validation loss: {results['best_val_loss']:.6f}")
    print(f"  Epochs trained:       {results['epochs_trained']}")

    print("\nGenerating visualizations...")
    visualize_results(results, parameters, coordinates, stress_fields, str(output_dir))


if __name__ == "__main__":
    main()
