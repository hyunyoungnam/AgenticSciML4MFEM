"""
Generate plate-with-two-holes mesh samples with natural blob-like hole shapes.

Each mesh contains two separate blob holes. Holes are guaranteed not to overlap
each other or the domain boundaries.

Physical group tags:
    bottom=1, right=2, top=3, left=4, hole1=5, hole2=6
"""

import os

import gmsh
import numpy as np
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
        amp = base_radius * np.random.uniform(0.08, 0.35) / np.sqrt(k)
        amplitudes.append(amp)
        phases.append(np.random.uniform(0, 2 * np.pi))

    # Elongation variation
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

        r = base_radius
        for k, amp, phase in zip(selected_modes, amplitudes, phases):
            r += amp * np.sin(k * theta + phase)

        for pos, width, height in zip(bump_positions, bump_widths, bump_heights):
            angular_dist = min(abs(theta - pos), 2 * np.pi - abs(theta - pos))
            r += height * np.exp(-angular_dist**2 / (2 * width**2))

        r = max(r, base_radius * 0.4)

        x_local = r * np.cos(theta)
        y_local = r * np.sin(theta)

        cos_e, sin_e = np.cos(elongation_angle), np.sin(elongation_angle)
        x_rot = x_local * cos_e + y_local * sin_e
        y_rot = -x_local * sin_e + y_local * cos_e
        x_rot *= aspect_ratio
        x_final = x_rot * cos_e - y_rot * sin_e
        y_final = x_rot * sin_e + y_rot * cos_e

        points.append((cx + x_final, cy + y_final))

    return points


def _blobs_overlap(cx1, cy1, r1, cx2, cy2, r2, min_gap=0.05):
    """Return True if two blobs (by center + radius) are too close."""
    dist = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
    return dist < (r1 + r2 + min_gap)


def create_plate_with_two_blob_holes(filename: str,
                                     center1: tuple, radius1: float,
                                     center2: tuple, radius2: float,
                                     mesh_size: float = 0.1,
                                     seed1: int = None, seed2: int = None):
    """
    Create a plate with two natural blob-like holes.

    Args:
        filename: output .mesh filename
        center1, radius1: center and base radius of first hole
        center2, radius2: center and base radius of second hole
        mesh_size: uniform mesh size
        seed1, seed2: random seeds for each hole shape
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("plate_two_blob_holes")

    # Outer square boundary [0,1] x [0,1]
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(1, 1, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, mesh_size)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Hole 1 spline
    blob1 = generate_blob_points(center1, radius1, n_points=48, seed=seed1)
    ids1 = [gmsh.model.geo.addPoint(x, y, 0, mesh_size) for x, y in blob1]
    spline1 = gmsh.model.geo.addSpline(ids1 + [ids1[0]])

    # Hole 2 spline
    blob2 = generate_blob_points(center2, radius2, n_points=48, seed=seed2)
    ids2 = [gmsh.model.geo.addPoint(x, y, 0, mesh_size) for x, y in blob2]
    spline2 = gmsh.model.geo.addSpline(ids2 + [ids2[0]])

    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    hole1_loop = gmsh.model.geo.addCurveLoop([spline1])
    hole2_loop = gmsh.model.geo.addCurveLoop([spline2])

    surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole1_loop, hole2_loop])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [l1], 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [l2], 2, "right")
    gmsh.model.addPhysicalGroup(1, [l3], 3, "top")
    gmsh.model.addPhysicalGroup(1, [l4], 4, "left")
    gmsh.model.addPhysicalGroup(1, [spline1], 5, "hole1")
    gmsh.model.addPhysicalGroup(1, [spline2], 6, "hole2")
    gmsh.model.addPhysicalGroup(2, [surface], 1, "domain")

    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()

    msh_file = filename.replace('.mesh', '.msh')
    gmsh.write(msh_file)
    gmsh.finalize()

    _convert_gmsh_to_mfem(msh_file, filename)
    os.remove(msh_file)


def _convert_gmsh_to_mfem(msh_file: str, mesh_file: str):
    """Convert Gmsh .msh file to MFEM .mesh format (supports quads and triangles)."""
    import meshio

    mesh = meshio.read(msh_file)

    points = mesh.points[:, :2]

    elements = []
    for cell_block in mesh.cells:
        if cell_block.type in ("quad", "quad8", "quad9"):
            for quad in cell_block.data:
                elements.append((3, quad[:4]))
        elif cell_block.type in ("triangle", "triangle6"):
            for tri in cell_block.data:
                elements.append((2, tri[:3]))

    if not elements:
        raise ValueError(
            f"No quads or triangles found; cell types present: "
            f"{[b.type for b in mesh.cells]}"
        )

    # Collect ALL boundary edges across every line cell block (one block per physical group)
    all_lines = []
    all_line_tags = []
    for idx, cell_block in enumerate(mesh.cells):
        if cell_block.type == "line":
            if "gmsh:physical" in mesh.cell_data:
                tags = mesh.cell_data["gmsh:physical"][idx]
            else:
                tags = [1] * len(cell_block.data)
            for j, line in enumerate(cell_block.data):
                all_lines.append(line)
                all_line_tags.append(int(tags[j]))

    with open(mesh_file, 'w') as f:
        f.write("MFEM mesh v1.0\n\n")
        f.write("dimension\n2\n\n")

        f.write(f"elements\n{len(elements)}\n")
        for elem_type, verts in elements:
            if elem_type == 3:
                f.write(f"1 3 {verts[0]} {verts[1]} {verts[2]} {verts[3]}\n")
            else:
                f.write(f"1 2 {verts[0]} {verts[1]} {verts[2]}\n")
        f.write("\n")

        # Boundary (lines) — all physical groups preserved
        if all_lines:
            f.write(f"boundary\n{len(all_lines)}\n")
            for line, tag in zip(all_lines, all_line_tags):
                f.write(f"{tag} 1 {line[0]} {line[1]}\n")
            f.write("\n")

        f.write(f"vertices\n{len(points)}\n2\n")
        for pt in points:
            f.write(f"{pt[0]} {pt[1]}\n")


def generate_all_two_hole_samples(output_dir: str, n_samples: int = 200,
                                   mesh_size: float = 0.1):
    """
    Generate n_samples plate-with-two-holes meshes.

    Args:
        output_dir: output directory
        n_samples: number of samples to generate
        mesh_size: uniform coarse mesh size
    """
    np.random.seed(42)

    os.makedirs(output_dir, exist_ok=True)

    for f in Path(output_dir).glob("sample_*.mesh"):
        os.remove(f)

    print(f"Generating {n_samples} two-hole plate samples...")
    print(f"Mesh size: {mesh_size}")

    for i in range(n_samples):
        # Per-hole radii: slightly smaller than single-hole to fit two
        r1 = 0.10 + 0.08 * np.random.random()
        r2 = 0.10 + 0.08 * np.random.random()

        # Sample hole positions, retrying if they overlap
        max_attempts = 30
        for attempt in range(max_attempts):
            m1 = r1 * 1.5 + 0.1
            cx1 = m1 + (1 - 2 * m1) * np.random.random()
            cy1 = m1 + (1 - 2 * m1) * np.random.random()

            m2 = r2 * 1.5 + 0.1
            cx2 = m2 + (1 - 2 * m2) * np.random.random()
            cy2 = m2 + (1 - 2 * m2) * np.random.random()

            if not _blobs_overlap(cx1, cy1, r1, cx2, cy2, r2, min_gap=0.05):
                break
        else:
            # Fallback: place holes in left/right halves
            cx1, cy1 = 0.25, 0.50
            cx2, cy2 = 0.75, 0.50

        filename = os.path.join(output_dir, f"sample_{i:03d}.mesh")

        # Try up to 5 different blob seeds if mesh generation fails
        generated = False
        for retry in range(5):
            try:
                # Seeds offset by 2000 to avoid overlap with train01 seeds (1000+i)
                create_plate_with_two_blob_holes(
                    filename,
                    center1=(cx1, cy1), radius1=r1,
                    center2=(cx2, cy2), radius2=r2,
                    mesh_size=mesh_size,
                    seed1=2000 + i + retry * 500,
                    seed2=3000 + i + retry * 500,
                )
                generated = True
                break
            except Exception as e:
                if retry == 4:
                    print(f"  [{i+1:3d}/{n_samples}] SKIPPED after 5 retries: {e}")
                continue

        if not generated:
            continue

        print(f"  [{i+1:3d}/{n_samples}] {Path(filename).name}  "
              f"hole1=(r={r1:.3f} @ {cx1:.2f},{cy1:.2f})  "
              f"hole2=(r={r2:.3f} @ {cx2:.2f},{cy2:.2f})")

    print(f"\nGenerated {n_samples} samples in {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate plate-with-two-holes mesh samples"
    )
    parser.add_argument("--n", type=int, default=200, help="Number of samples")
    parser.add_argument("--mesh-size", type=float, default=0.1,
                        help="Coarse mesh size (default: 0.1)")
    parser.add_argument("--output", type=str, default="train02",
                        help="Output directory (default: train02)")

    args = parser.parse_args()

    out = args.output if os.path.isabs(args.output) else \
        str(Path(__file__).parent.parent / args.output)

    generate_all_two_hole_samples(out, n_samples=args.n, mesh_size=args.mesh_size)
