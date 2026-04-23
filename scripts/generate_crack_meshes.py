"""
Generate crack mesh files for training.

Creates a small set of edge crack meshes with varying parameters:
- Crack length (a/W ratio)
- Crack angle
- Resolution

Output: crack_data/ directory with MFEM mesh files
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from piano.geometry import generate_crack_mesh, EdgeCrack, CrackMeshGenerator


def generate_training_meshes(
    output_dir: str = "crack_data",
    n_samples: int = 10,
    seed: int = 42,
):
    """
    Generate crack mesh files for training.

    Parameters varied:
    - crack_length: 0.2 to 0.5 (a/W ratio of 0.2 to 0.5)
    - crack_angle: -15 to +15 degrees (for mixed mode)
    - crack_y: 0.4 to 0.6 (vertical position variation)

    Args:
        output_dir: Directory to save mesh files
        n_samples: Number of samples to generate
        seed: Random seed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    print(f"Generating {n_samples} crack meshes in {output_path}/")
    print("=" * 50)

    metadata_list = []

    for i in range(n_samples):
        # Sample parameters
        crack_length = float(rng.uniform(0.2, 0.5))  # a/W ratio
        crack_angle = float(rng.uniform(-15, 15))    # degrees
        crack_y = float(rng.uniform(0.4, 0.6))       # vertical position

        # Higher resolution for longer cracks
        resolution = int(20 + 10 * (crack_length / 0.5))

        mesh_file = output_path / f"crack_{i:03d}.mesh"

        try:
            vertices, elements, meta = generate_crack_mesh(
                crack_type="edge",
                crack_length=crack_length,
                crack_angle=crack_angle,
                crack_y=crack_y,
                width=1.0,
                height=1.0,
                resolution=resolution,
                tip_refinement=3,
                tip_radius=0.15,
                output_path=str(mesh_file),
            )

            meta.update({
                "sample_id": i,
                "crack_length": crack_length,
                "crack_angle": crack_angle,
                "crack_y": crack_y,
            })
            metadata_list.append(meta)

            print(f"  [{i+1:3d}/{n_samples}] crack_{i:03d}.mesh: "
                  f"a={crack_length:.3f}, angle={crack_angle:+.1f}deg, "
                  f"y={crack_y:.2f}, {meta['n_vertices']} verts, {meta['n_elements']} elems")

        except Exception as e:
            print(f"  [{i+1:3d}/{n_samples}] FAILED: {e}")

    # Save metadata
    import json
    meta_file = output_path / "metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata_list, f, indent=2)

    print("=" * 50)
    print(f"Generated {len(metadata_list)} meshes")
    print(f"Metadata saved to {meta_file}")

    return metadata_list


def visualize_sample_mesh(mesh_idx: int = 0, data_dir: str = "crack_data"):
    """Visualize a sample crack mesh."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import json

    data_path = Path(data_dir)

    # Load metadata
    with open(data_path / "metadata.json") as f:
        metadata = json.load(f)

    if mesh_idx >= len(metadata):
        print(f"Only {len(metadata)} meshes available")
        return

    meta = metadata[mesh_idx]

    # Load mesh (parse MFEM format)
    mesh_file = data_path / f"crack_{mesh_idx:03d}.mesh"
    vertices, elements = _parse_mfem_mesh(mesh_file)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], elements)
    ax.triplot(triang, 'b-', lw=0.5)

    # Mark crack tip
    tip_pos = meta["tip_positions"][0]
    ax.plot(tip_pos[0], tip_pos[1], 'r*', markersize=15, label='Crack tip')

    # Mark crack path (from left edge to tip)
    crack_y = meta.get("crack_y", 0.5)
    ax.plot([0, tip_pos[0]], [crack_y, tip_pos[1]], 'r-', lw=2, label='Crack')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title(f"Crack Mesh: a={meta['crack_length']:.3f}, "
                 f"angle={meta['crack_angle']:.1f}deg\n"
                 f"{meta['n_vertices']} vertices, {meta['n_elements']} triangles")
    ax.legend()

    output_file = data_path / f"crack_{mesh_idx:03d}_viz.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_file}")


def _parse_mfem_mesh(mesh_file: Path):
    """Parse MFEM mesh file to get vertices and elements."""
    vertices = []
    elements = []

    with open(mesh_file) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "elements":
            n_elem = int(lines[i + 1].strip())
            for j in range(n_elem):
                parts = lines[i + 2 + j].strip().split()
                # Format: attr type v1 v2 v3
                elements.append([int(parts[2]), int(parts[3]), int(parts[4])])
            i += 2 + n_elem

        elif line == "vertices":
            n_vert = int(lines[i + 1].strip())
            dim = int(lines[i + 2].strip())
            for j in range(n_vert):
                parts = lines[i + 3 + j].strip().split()
                vertices.append([float(parts[0]), float(parts[1])])
            i += 3 + n_vert

        else:
            i += 1

    return np.array(vertices), np.array(elements)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate crack meshes for training")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of mesh samples (default: 10)")
    parser.add_argument("--output-dir", type=str, default="crack_data",
                        help="Output directory (default: crack_data)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--visualize", type=int, default=None,
                        help="Visualize mesh at given index after generation")
    args = parser.parse_args()

    generate_training_meshes(
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    if args.visualize is not None:
        visualize_sample_mesh(args.visualize, args.output_dir)
