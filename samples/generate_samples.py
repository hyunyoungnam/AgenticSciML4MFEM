"""
Generate 30 plate-with-hole mesh samples with natural blob-like hole shapes.

The holes have smooth, organic perimeters similar to natural inclusions,
created using low-frequency Fourier modes with slight elongation.

These are BASE meshes with uniform coarse resolution - refinement happens
during the SciML loop.
"""

import gmsh
import numpy as np
import os
from pathlib import Path


def generate_blob_points(center: tuple, base_radius: float, n_points: int = 48,
                         seed: int = None):
    """
    Generate points for a natural blob/potato-like shape with high variability.

    Uses randomized Fourier modes and local perturbations to create diverse,
    organic shapes similar to natural holes or inclusions.

    Args:
        center: (cx, cy) center of the shape
        base_radius: base radius before perturbation
        n_points: number of points to generate on the perimeter
        seed: random seed for reproducibility

    Returns:
        List of (x, y) coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    cx, cy = center

    # Randomly select which modes to use (creates more variety)
    available_modes = [2, 3, 4, 5, 6, 7]
    n_modes = np.random.randint(2, 5)  # Use 2-4 modes
    selected_modes = np.random.choice(available_modes, size=n_modes, replace=False)

    # Random amplitudes - some shapes will be more deformed than others
    amplitudes = []
    phases = []
    for k in selected_modes:
        # Higher variance in amplitudes for more diversity
        amp = base_radius * np.random.uniform(0.08, 0.35) / np.sqrt(k)
        amplitudes.append(amp)
        phases.append(np.random.uniform(0, 2 * np.pi))

    # Stronger elongation variation
    aspect_ratio = np.random.uniform(0.6, 1.4)
    elongation_angle = np.random.uniform(0, 2 * np.pi)

    # Add random local bumps/indentations
    n_bumps = np.random.randint(0, 4)
    bump_positions = np.random.uniform(0, 2 * np.pi, n_bumps)
    bump_widths = np.random.uniform(0.3, 0.8, n_bumps)
    bump_heights = np.random.uniform(-0.15, 0.2, n_bumps) * base_radius

    points = []
    for i in range(n_points):
        theta = 2 * np.pi * i / n_points

        # Base radius with Fourier mode perturbations
        r = base_radius
        for k, amp, phase in zip(selected_modes, amplitudes, phases):
            r += amp * np.sin(k * theta + phase)

        # Add local bumps (Gaussian-shaped)
        for pos, width, height in zip(bump_positions, bump_widths, bump_heights):
            angular_dist = min(abs(theta - pos), 2 * np.pi - abs(theta - pos))
            r += height * np.exp(-angular_dist**2 / (2 * width**2))

        # Ensure radius stays positive
        r = max(r, base_radius * 0.4)

        # Apply elongation
        x_local = r * np.cos(theta)
        y_local = r * np.sin(theta)

        # Rotate to elongation direction, scale, rotate back
        cos_e, sin_e = np.cos(elongation_angle), np.sin(elongation_angle)
        x_rot = x_local * cos_e + y_local * sin_e
        y_rot = -x_local * sin_e + y_local * cos_e
        x_rot *= aspect_ratio
        x_final = x_rot * cos_e - y_rot * sin_e
        y_final = x_rot * sin_e + y_rot * cos_e

        points.append((cx + x_final, cy + y_final))

    return points


def create_plate_with_blob_hole(filename: str, center: tuple, base_radius: float,
                                 mesh_size: float = 0.1, seed: int = None):
    """
    Create a plate with a natural blob-like hole.

    Args:
        filename: output .mesh filename
        center: (cx, cy) hole center
        base_radius: base radius of the hole
        mesh_size: uniform mesh size (coarse for base mesh)
        seed: random seed for shape reproducibility
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)  # Suppress output
    gmsh.model.add("plate_blob_hole")

    # Outer square boundary [0,1] x [0,1]
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(1, 1, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, mesh_size)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Generate blob hole points - use SAME mesh_size (no refinement near hole)
    blob_points = generate_blob_points(
        center, base_radius,
        n_points=48,
        seed=seed
    )

    hole_point_ids = []
    for x, y in blob_points:
        hole_point_ids.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))

    # Create smooth spline for the amoeba boundary
    hole_spline = gmsh.model.geo.addSpline(hole_point_ids + [hole_point_ids[0]])

    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    hole_loop = gmsh.model.geo.addCurveLoop([hole_spline])

    surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])

    gmsh.model.geo.synchronize()

    # Set boundary attributes
    gmsh.model.addPhysicalGroup(1, [l1], 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [l2], 2, "right")
    gmsh.model.addPhysicalGroup(1, [l3], 3, "top")
    gmsh.model.addPhysicalGroup(1, [l4], 4, "left")
    gmsh.model.addPhysicalGroup(1, [hole_spline], 5, "hole")
    gmsh.model.addPhysicalGroup(2, [surface], 1, "domain")

    # Use quad mesh instead of triangles
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)  # simple full-quad

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()

    # Save as msh and convert to MFEM
    msh_file = filename.replace('.mesh', '.msh')
    gmsh.write(msh_file)
    gmsh.finalize()

    convert_gmsh_to_mfem(msh_file, filename)
    os.remove(msh_file)


def convert_gmsh_to_mfem(msh_file: str, mesh_file: str):
    """Convert Gmsh .msh file to MFEM .mesh format (supports quads and triangles)."""
    import meshio

    mesh = meshio.read(msh_file)

    # Extract 2D elements (quads preferred, triangles as fallback)
    points = mesh.points[:, :2]  # Only x, y coordinates

    elements = []  # List of (type, vertices) where type is 2=tri, 3=quad

    for cell_block in mesh.cells:
        if cell_block.type == "quad":
            for quad in cell_block.data:
                elements.append((3, quad))  # MFEM type 3 = SQUARE/QUAD
        elif cell_block.type == "triangle":
            for tri in cell_block.data:
                elements.append((2, tri))  # MFEM type 2 = TRIANGLE

    if not elements:
        raise ValueError("No quads or triangles found in mesh")

    # Extract boundary edges (lines)
    lines = None
    line_tags = None
    for cell_block in mesh.cells:
        if cell_block.type == "line":
            lines = cell_block.data
            break

    # Get physical tags for boundaries
    if "gmsh:physical" in mesh.cell_data:
        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type == "line":
                line_tags = mesh.cell_data["gmsh:physical"][i]
                break

    # Write MFEM format
    with open(mesh_file, 'w') as f:
        f.write("MFEM mesh v1.0\n\n")
        f.write("dimension\n2\n\n")

        # Elements (quads and/or triangles)
        f.write(f"elements\n{len(elements)}\n")
        for elem_type, verts in elements:
            if elem_type == 3:  # Quad
                f.write(f"1 3 {verts[0]} {verts[1]} {verts[2]} {verts[3]}\n")
            else:  # Triangle
                f.write(f"1 2 {verts[0]} {verts[1]} {verts[2]}\n")
        f.write("\n")

        # Boundary (lines)
        if lines is not None:
            f.write(f"boundary\n{len(lines)}\n")
            for i, line in enumerate(lines):
                tag = line_tags[i] if line_tags is not None else 1
                f.write(f"{tag} 1 {line[0]} {line[1]}\n")
            f.write("\n")

        # Vertices
        f.write(f"vertices\n{len(points)}\n2\n")
        for pt in points:
            f.write(f"{pt[0]} {pt[1]}\n")


def generate_all_samples(output_dir: str, n_samples: int = 30, mesh_size: float = 0.1):
    """
    Generate n_samples plate-with-hole meshes with natural blob-like shapes.

    Args:
        output_dir: output directory
        n_samples: number of samples to generate
        mesh_size: uniform coarse mesh size (default 0.1 for base mesh)
    """
    np.random.seed(42)  # For reproducibility

    os.makedirs(output_dir, exist_ok=True)

    # Clear existing sample files
    for f in Path(output_dir).glob("sample_*.mesh"):
        os.remove(f)

    print(f"Generating {n_samples} blob-shaped hole samples...")
    print(f"Mesh size: {mesh_size} (coarse base mesh)")

    for i in range(n_samples):
        # Vary parameters for diversity

        # Base radius: 0.12 to 0.22 (similar to reference image)
        base_radius = 0.12 + 0.1 * np.random.random()

        # Safe center position (keep hole away from boundaries)
        margin = base_radius * 1.5 + 0.1
        cx = margin + (1 - 2 * margin) * np.random.random()
        cy = margin + (1 - 2 * margin) * np.random.random()

        filename = os.path.join(output_dir, f"sample_{i:03d}.mesh")

        create_plate_with_blob_hole(
            filename,
            center=(cx, cy),
            base_radius=base_radius,
            mesh_size=mesh_size,
            seed=1000 + i  # Reproducible but unique per sample
        )

        print(f"  [{i+1:3d}/{n_samples}] Created {Path(filename).name} "
              f"(r={base_radius:.3f})")

    print(f"\nGenerated {n_samples} samples in {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate plate-with-hole mesh samples")
    parser.add_argument("--n", type=int, default=30, help="Number of samples")
    parser.add_argument("--mesh-size", type=float, default=0.1,
                        help="Coarse mesh size (default: 0.1)")
    parser.add_argument("--output", type=str, default=".", help="Output directory")

    args = parser.parse_args()

    script_dir = Path(__file__).parent if args.output == "." else Path(args.output)
    generate_all_samples(str(script_dir), n_samples=args.n, mesh_size=args.mesh_size)
