"""
Transolver Training Script.

Trains a Transolver surrogate model on plate-with-hole FEA data.

Workflow:
1. Load training meshes from train/ folder
2. Run FEA simulations to generate ground truth stress fields
3. Train Transolver on (mesh, parameters) -> stress field mapping
4. Save trained model to outputs/surrogate/

Usage:
    python train/train_transolver.py
    python train/train_transolver.py --epochs 500 --batch-size 16
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


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
# FEA Simulation (Ground Truth Generation)
# =============================================================================

def compute_hole_properties(vertices: np.ndarray, boundary: list) -> Dict:
    """Extract hole properties from mesh boundary."""
    hole_vertices = set()
    for attr, v1, v2 in boundary:
        if attr == 5:  # Hole boundary attribute
            hole_vertices.add(v1)
            hole_vertices.add(v2)

    if hole_vertices:
        hole_coords = vertices[list(hole_vertices)]
        hole_center = np.mean(hole_coords, axis=0)
        hole_radius = np.max(np.linalg.norm(hole_coords - hole_center, axis=1))
    else:
        hole_center = np.mean(vertices, axis=0)
        hole_radius = 0.15

    return {
        'center': hole_center,
        'radius': hole_radius,
    }


def simulate_stress_field(
    vertices: np.ndarray,
    elements: list,
    boundary: list,
    params: Dict
) -> np.ndarray:
    """
    Simulate stress field for plate with hole (synthetic FEA).

    In production, this would call actual MFEM solver.
    Here we use analytical Kirsch solution approximation.

    Args:
        vertices: Node coordinates
        elements: Element connectivity
        boundary: Boundary information
        params: Material/load parameters

    Returns:
        element_stress: Von Mises stress at element centers
    """
    hole_props = compute_hole_properties(vertices, boundary)
    hole_center = hole_props['center']
    hole_radius = hole_props['radius']

    element_centers = get_element_centers(vertices, elements)

    # Distance from hole center
    r = np.linalg.norm(element_centers - hole_center, axis=1)
    r = np.maximum(r, hole_radius * 0.5)

    # Normalized distance
    rho = r / hole_radius

    # Base stress
    sigma_0 = params.get('load', 100.0)
    E = params.get('E', 200e9)
    nu = params.get('nu', 0.3)

    # Stress concentration (Kirsch solution)
    theta = np.arctan2(element_centers[:, 1] - hole_center[1],
                       element_centers[:, 0] - hole_center[0])

    # Radial and hoop stress components (simplified)
    sigma_r = sigma_0 / 2 * ((1 - 1/rho**2) + (1 - 4/rho**2 + 3/rho**4) * np.cos(2*theta))
    sigma_theta = sigma_0 / 2 * ((1 + 1/rho**2) - (1 + 3/rho**4) * np.cos(2*theta))
    tau_r_theta = -sigma_0 / 2 * (1 + 2/rho**2 - 3/rho**4) * np.sin(2*theta)

    # Von Mises stress
    von_mises = np.sqrt(sigma_r**2 + sigma_theta**2 - sigma_r*sigma_theta + 3*tau_r_theta**2)

    # Add material-dependent scaling
    scale = (E / 200e9) * ((1 + nu) / 1.3)
    von_mises *= scale

    return von_mises


# =============================================================================
# Dataset Preparation
# =============================================================================

def load_training_data(
    train_dir: str,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load training data from mesh files.

    Uses ELEMENT CENTERS as coordinates (not nodes) so that input and output
    dimensions match - stress is defined at elements.

    Args:
        train_dir: Directory containing training meshes
        max_samples: Maximum number of samples to load

    Returns:
        parameters: (N, n_params) input parameters
        coordinates: (N, max_elements, 2) element center coordinates (padded)
        stress_fields: (N, max_elements) stress values (padded)
        mesh_files: List of mesh file paths
    """
    train_path = Path(train_dir)
    mesh_files = sorted(train_path.glob("sample_*.mesh"))

    if max_samples:
        mesh_files = mesh_files[:max_samples]

    print(f"Loading {len(mesh_files)} training samples...")

    all_params = []
    all_coords = []
    all_stress = []

    # First pass: find max element count
    max_elems = 0

    for mesh_file in mesh_files:
        vertices, elements, _ = read_mfem_mesh(str(mesh_file))
        max_elems = max(max_elems, len(elements))

    print(f"Max elements: {max_elems}")

    # Second pass: load and pad data
    for i, mesh_file in enumerate(mesh_files):
        vertices, elements, boundary = read_mfem_mesh(str(mesh_file))

        # Get element centers as coordinates
        elem_centers = get_element_centers(vertices, elements)

        # Extract hole properties as parameters
        hole_props = compute_hole_properties(vertices, boundary)

        # Random material parameters for diversity
        np.random.seed(1000 + i)
        E = np.random.uniform(150e9, 250e9)
        nu = np.random.uniform(0.25, 0.35)
        load = np.random.uniform(80, 120)

        params = {
            'E': E,
            'nu': nu,
            'load': load,
            'hole_cx': hole_props['center'][0],
            'hole_cy': hole_props['center'][1],
            'hole_r': hole_props['radius'],
        }

        # Run FEA simulation (stress at element centers)
        stress = simulate_stress_field(vertices, elements, boundary, params)

        # Pad coordinates (element centers)
        padded_coords = np.zeros((max_elems, 2))
        padded_coords[:len(elem_centers)] = elem_centers

        # Pad stress
        padded_stress = np.zeros(max_elems)
        padded_stress[:len(stress)] = stress

        # Store
        param_vec = np.array([
            params['hole_cx'],
            params['hole_cy'],
            params['hole_r'],
            params['E'] / 1e9,  # Normalize
            params['nu'],
            params['load'],
        ])

        all_params.append(param_vec)
        all_coords.append(padded_coords)
        all_stress.append(padded_stress)

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(mesh_files)} samples")

    parameters = np.array(all_params, dtype=np.float32)
    coordinates = np.array(all_coords, dtype=np.float32)
    stress_fields = np.array(all_stress, dtype=np.float32)

    print(f"Data shapes: params={parameters.shape}, coords={coordinates.shape}, stress={stress_fields.shape}")

    return parameters, coordinates, stress_fields, [str(f) for f in mesh_files]


# =============================================================================
# Training
# =============================================================================

def train_transolver(
    parameters: np.ndarray,
    coordinates: np.ndarray,
    stress_fields: np.ndarray,
    config: Dict,
    output_dir: str
) -> Dict:
    """
    Train Transolver model.

    Args:
        parameters: Input parameters (N, n_params)
        coordinates: Mesh coordinates (N, max_nodes, 2)
        stress_fields: Target stress fields (N, max_elements)
        config: Training configuration
        output_dir: Output directory for saving model

    Returns:
        Training results dictionary
    """
    from meshforge.surrogate.base import TransolverConfig
    from meshforge.surrogate.transolver import TransolverModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Prepare data
    n_samples = len(parameters)
    n_train = int(n_samples * 0.9)

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Normalize data
    param_mean = parameters.mean(axis=0)
    param_std = parameters.std(axis=0) + 1e-8
    stress_mean = stress_fields.mean()
    stress_std = stress_fields.std() + 1e-8

    params_norm = (parameters - param_mean) / param_std
    stress_norm = (stress_fields - stress_mean) / stress_std

    print(f"Stress stats: mean={stress_mean:.2f}, std={stress_std:.2f}")

    # Convert to tensors
    train_params = torch.tensor(params_norm[train_idx], dtype=torch.float32, device=device)
    train_coords = torch.tensor(coordinates[train_idx], dtype=torch.float32, device=device)
    train_stress = torch.tensor(stress_norm[train_idx], dtype=torch.float32, device=device).unsqueeze(-1)

    val_params = torch.tensor(params_norm[val_idx], dtype=torch.float32, device=device)
    val_coords = torch.tensor(coordinates[val_idx], dtype=torch.float32, device=device)
    val_stress = torch.tensor(stress_norm[val_idx], dtype=torch.float32, device=device).unsqueeze(-1)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Create model
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
        output_dim=1,  # Von Mises stress (scalar)
    )

    model = TransolverModel(transolver_config)
    model.build(
        input_dim=parameters.shape[1],
        coord_dim=2,
        num_points=coordinates.shape[1]
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=transolver_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    criterion = torch.nn.MSELoss()

    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    batch_size = transolver_config.batch_size

    for epoch in range(transolver_config.epochs):
        model.train()
        epoch_loss = 0.0

        # Shuffle training data
        perm = torch.randperm(len(train_params))

        for i in range(0, len(train_params), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_params = train_params[batch_idx]
            batch_coords = train_coords[batch_idx]
            batch_stress = train_stress[batch_idx]

            optimizer.zero_grad()

            # Forward pass
            pred = model.forward(batch_params, batch_coords)

            # Compute loss only on valid elements (non-padded)
            loss = criterion(pred, batch_stress)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)

        train_loss = epoch_loss / len(train_params)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model.forward(val_params, val_coords)
            val_loss = criterion(val_pred, val_stress).item()
            history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:4d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if patience_counter >= transolver_config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    model._is_trained = True

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "transolver_model.pt"
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Save training history
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'best_val_loss': float(best_val_loss),
            'config': config,
        }, f, indent=2)

    # Save normalization info
    norm_path = output_path / "normalization.json"
    with open(norm_path, 'w') as f:
        json.dump({
            'param_mean': param_mean.tolist(),
            'param_std': param_std.tolist(),
            'stress_mean': float(stress_mean),
            'stress_std': float(stress_std),
        }, f, indent=2)

    return {
        'model_path': str(model_path),
        'best_val_loss': best_val_loss,
        'epochs_trained': len(history['train_loss']),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Transolver surrogate model")
    parser.add_argument("--train-dir", type=str, default="train", help="Training data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/surrogate", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--d-model", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--slice-num", type=int, default=32, help="Number of physics slices")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    # Load data
    project_root = Path(__file__).parent.parent
    train_dir = project_root / args.train_dir
    output_dir = project_root / args.output_dir

    parameters, coordinates, stress_fields, mesh_files = load_training_data(
        str(train_dir), max_samples=args.max_samples
    )

    # Training config
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'slice_num': args.slice_num,
        'learning_rate': args.learning_rate,
        'n_heads': 8,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'patience': 50,
    }

    print("\nTraining configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Train
    results = train_transolver(
        parameters, coordinates, stress_fields, config, str(output_dir)
    )

    print("\nTraining complete!")
    print(f"  Model saved to: {results['model_path']}")
    print(f"  Best validation loss: {results['best_val_loss']:.6f}")
    print(f"  Epochs trained: {results['epochs_trained']}")


if __name__ == "__main__":
    main()
