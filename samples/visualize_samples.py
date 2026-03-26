"""
Visualize plate-with-hole mesh samples using matplotlib.

Usage:
    python visualize_samples.py                    # Grid view of all samples
    python visualize_samples.py --mode bytype      # Grouped by hole type
    python visualize_samples.py --file sample.mesh # Single file
    python visualize_samples.py --no-show          # Save only, don't display
"""

import numpy as np
import matplotlib
import argparse
import sys

# Parse --no-show early to set backend before importing pyplot
if "--no-show" in sys.argv:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pathlib import Path
import argparse


def read_mfem_mesh(filename: str):
    """Read an MFEM mesh file and return vertices and elements."""
    vertices = []
    elements = []

    with open(filename, 'r') as f:
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
                # Format: attr type v1 v2 v3 [v4]
                elem_type = int(parts[1])
                if elem_type == 2:  # Triangle
                    elements.append([int(parts[2]), int(parts[3]), int(parts[4])])
                elif elem_type == 3:  # Quad - reorder for proper polygon rendering
                    # MFEM quad order: v0, v1, v2, v3 (counterclockwise)
                    elements.append([int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])])
                i += 1

        elif line == "vertices":
            i += 1
            n_vertices = int(lines[i].strip())
            i += 1
            dim = int(lines[i].strip())
            i += 1
            for _ in range(n_vertices):
                coords = [float(x) for x in lines[i].strip().split()]
                vertices.append(coords[:2])  # Only x, y
                i += 1
        else:
            i += 1

    return np.array(vertices), elements


def plot_mesh(ax, vertices, elements, title="", show_edges=True):
    """Plot a 2D mesh on the given axes."""
    # Create polygon collection
    polygons = []
    for elem in elements:
        poly = vertices[elem]
        polygons.append(poly)

    collection = PolyCollection(
        polygons,
        edgecolors='black' if show_edges else 'none',
        facecolors='lightblue',
        linewidths=0.3,
        alpha=0.7
    )
    ax.add_collection(collection)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def visualize_single(mesh_file: str):
    """Visualize a single mesh file."""
    vertices, elements = read_mfem_mesh(mesh_file)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_mesh(ax, vertices, elements, title=Path(mesh_file).stem)
    plt.tight_layout()
    plt.show()


def visualize_grid(mesh_dir: str, n_cols: int = 5, max_samples: int = 50):
    """Visualize multiple mesh files in a grid."""
    mesh_files = sorted(Path(mesh_dir).glob("sample_*.mesh"))[:max_samples]

    if not mesh_files:
        print(f"No mesh files found in {mesh_dir}")
        return

    n_files = len(mesh_files)
    n_rows = (n_files + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()

    for i, mesh_file in enumerate(mesh_files):
        try:
            vertices, elements = read_mfem_mesh(str(mesh_file))
            plot_mesh(axes[i], vertices, elements, title=mesh_file.stem, show_edges=True)
        except Exception as e:
            print(f"Error loading {mesh_file}: {e}")
            axes[i].set_title(f"Error: {mesh_file.stem}")

    # Hide empty subplots
    for i in range(n_files, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(Path(mesh_dir) / "samples_overview.png", dpi=150)
    print(f"Saved overview to {Path(mesh_dir) / 'samples_overview.png'}")
    plt.show()


def visualize_by_type(mesh_dir: str):
    """Visualize samples grouped by hole type."""
    mesh_files = sorted(Path(mesh_dir).glob("sample_*.mesh"))

    # Group by type
    types = {}
    for f in mesh_files:
        hole_type = f.stem.split('_')[-1]
        if hole_type not in types:
            types[hole_type] = []
        types[hole_type].append(f)

    fig, axes = plt.subplots(len(types), 5, figsize=(15, 3 * len(types)))

    for row, (hole_type, files) in enumerate(sorted(types.items())):
        for col in range(5):
            ax = axes[row, col] if len(types) > 1 else axes[col]
            if col < len(files):
                try:
                    vertices, elements = read_mfem_mesh(str(files[col]))
                    title = f"{hole_type}" if col == 0 else ""
                    plot_mesh(ax, vertices, elements, title=title)
                except Exception as e:
                    ax.set_title(f"Error")
            else:
                ax.axis('off')

        # Add row label
        ax_first = axes[row, 0] if len(types) > 1 else axes[0]
        ax_first.set_ylabel(hole_type, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(Path(mesh_dir) / "samples_by_type.png", dpi=150)
    print(f"Saved by-type view to {Path(mesh_dir) / 'samples_by_type.png'}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize plate-with-hole mesh samples")
    parser.add_argument("--file", type=str, help="Single mesh file to visualize")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing mesh files")
    parser.add_argument("--mode", choices=["single", "grid", "bytype"], default="grid",
                        help="Visualization mode")
    parser.add_argument("--cols", type=int, default=10, help="Number of columns in grid view")
    parser.add_argument("--no-show", action="store_true", help="Save images without displaying")

    args = parser.parse_args()

    if args.file:
        visualize_single(args.file)
    elif args.mode == "grid":
        visualize_grid(args.dir, n_cols=args.cols)
    elif args.mode == "bytype":
        visualize_by_type(args.dir)
