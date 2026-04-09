"""
Generate fine mesh samples using scipy Delaunay triangulation.

No gmsh or system GL libraries required.  Writes MFEM .mesh v1.0 directly.

Local refinement is achieved by placing a fine point grid within
`refinement_depth` of the hole boundary, with a coarser grid elsewhere.

Typical usage:
    python3 samples/generate_fine_samples.py                  # 200 meshes → train_fine/
    python3 samples/generate_fine_samples.py --n 50 --output train_fine_small
    python3 samples/generate_fine_samples.py \\
        --mesh-size-far 0.06 --mesh-size-hole 0.018 --n 200

Typical element counts:
    mesh_size_far=0.07, mesh_size_hole=0.025 → ~400-600 triangles per mesh
    mesh_size_far=0.06, mesh_size_hole=0.018 → ~700-1000 triangles per mesh
"""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# Blob / hole shape generation  (same logic as generate_samples.py)
# ---------------------------------------------------------------------------

def generate_blob_points(
    center: tuple,
    base_radius: float,
    n_points: int = 48,
    seed: int = None,
) -> np.ndarray:
    """Return (n_points, 2) array of blob-hole boundary coordinates."""
    if seed is not None:
        np.random.seed(seed)

    cx, cy = center
    available_modes = [2, 3, 4, 5, 6, 7]
    n_modes       = np.random.randint(2, 5)
    selected_modes = np.random.choice(available_modes, size=n_modes, replace=False)

    amplitudes, phases = [], []
    for k in selected_modes:
        amplitudes.append(base_radius * np.random.uniform(0.08, 0.35) / np.sqrt(k))
        phases.append(np.random.uniform(0, 2 * np.pi))

    aspect_ratio     = np.random.uniform(0.6, 1.4)
    elongation_angle = np.random.uniform(0, 2 * np.pi)

    n_bumps       = np.random.randint(0, 4)
    bump_positions = np.random.uniform(0, 2 * np.pi, n_bumps)
    bump_widths    = np.random.uniform(0.3, 0.8, n_bumps)
    bump_heights   = np.random.uniform(-0.15, 0.2, n_bumps) * base_radius

    pts = []
    for i in range(n_points):
        theta = 2 * np.pi * i / n_points
        r = base_radius
        for k, amp, phase in zip(selected_modes, amplitudes, phases):
            r += amp * np.sin(k * theta + phase)
        for pos, width, height in zip(bump_positions, bump_widths, bump_heights):
            ang_dist = min(abs(theta - pos), 2 * np.pi - abs(theta - pos))
            r += height * np.exp(-ang_dist ** 2 / (2 * width ** 2))
        r = max(r, base_radius * 0.4)

        x_local = r * np.cos(theta)
        y_local = r * np.sin(theta)
        cos_e, sin_e = np.cos(elongation_angle), np.sin(elongation_angle)
        x_rot = x_local * cos_e + y_local * sin_e
        y_rot = -x_local * sin_e + y_local * cos_e
        x_rot *= aspect_ratio
        x_final = x_rot * cos_e - y_rot * sin_e
        y_final = x_rot * sin_e + y_rot * cos_e
        pts.append([cx + x_final, cy + y_final])

    return np.array(pts)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y   = pt
    inside = False
    j      = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def dist_to_polygon_boundary(pt: np.ndarray, poly: np.ndarray) -> float:
    """Minimum distance from pt to any edge of the polygon."""
    min_d = np.inf
    n = len(poly)
    for i in range(n):
        a  = poly[i]
        b  = poly[(i + 1) % n]
        ab = b - a
        ap = pt - a
        t  = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-14), 0.0, 1.0)
        d  = np.linalg.norm(pt - (a + t * ab))
        if d < min_d:
            min_d = d
    return min_d


# ---------------------------------------------------------------------------
# Point cloud generation
# ---------------------------------------------------------------------------

def _make_boundary_pts(mesh_size: float) -> np.ndarray:
    """Uniform points on the outer square [0,1]^2 edges."""
    n   = max(3, int(round(1.0 / mesh_size)))
    ts  = np.linspace(0, 1, n + 1)
    pts = []
    for t in ts:
        pts.extend([[t, 0], [t, 1], [0, t], [1, t]])
    return np.unique(np.array(pts), axis=0)


def _make_interior_pts(
    hole_poly: np.ndarray,
    mesh_size_far: float,
    mesh_size_hole: float,
    refinement_depth: float,
    jitter: float = 0.3,
) -> np.ndarray:
    """
    Interior (non-boundary) points with local refinement near the hole.

    Points within `refinement_depth` of the hole boundary use
    `mesh_size_hole` spacing; all others use `mesh_size_far`.
    """
    rng = np.random.default_rng(0)

    # Pre-compute hole bounding box to accelerate the distance query
    bb_lo = hole_poly.min(axis=0) - refinement_depth - 0.01
    bb_hi = hole_poly.max(axis=0) + refinement_depth + 0.01
    bb_lo = np.maximum(bb_lo, 0.005)
    bb_hi = np.minimum(bb_hi, 0.995)

    pts = []

    # Safe interior margin — interior points must never reach the outer boundary
    # (that would corrupt the convex hull and mis-tag boundary edges).
    safe_lo = 0.5 * mesh_size_far
    safe_hi = 1.0 - safe_lo

    def _jitter_clamp(x: float, y: float, h: float) -> list:
        """Apply jitter then clamp to safe interior box."""
        jx = rng.uniform(-jitter * h, jitter * h)
        jy = rng.uniform(-jitter * h, jitter * h)
        return [
            float(np.clip(x + jx, safe_lo, safe_hi)),
            float(np.clip(y + jy, safe_lo, safe_hi)),
        ]

    # ── Fine grid near hole ──────────────────────────────────────────────
    xs = np.arange(bb_lo[0], bb_hi[0] + mesh_size_hole * 0.5, mesh_size_hole)
    ys = np.arange(bb_lo[1], bb_hi[1] + mesh_size_hole * 0.5, mesh_size_hole)
    for x in xs:
        for y in ys:
            p   = np.array([x, y])
            d   = dist_to_polygon_boundary(p, hole_poly)
            if d > refinement_depth:
                continue                   # handled by coarse grid
            if point_in_polygon(p, hole_poly):
                continue
            pts.append(_jitter_clamp(x, y, mesh_size_hole))

    # ── Coarse grid outside refinement zone ─────────────────────────────
    margin = mesh_size_far * 0.5
    xs_c = np.arange(margin, 1.0, mesh_size_far)
    ys_c = np.arange(margin, 1.0, mesh_size_far)
    for x in xs_c:
        for y in ys_c:
            p = np.array([x, y])
            if point_in_polygon(p, hole_poly):
                continue
            d = dist_to_polygon_boundary(p, hole_poly)
            if d <= refinement_depth:
                continue                   # already covered above
            pts.append(_jitter_clamp(x, y, mesh_size_far))

    return np.array(pts) if pts else np.empty((0, 2))


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

def _triangulate_and_cull(pts: np.ndarray, hole_poly: np.ndarray):
    """
    Delaunay triangulate pts; remove triangles whose centroid is inside hole.

    Returns (verts, triangles) where verts is (NV, 2) and triangles is
    a list of [v0, v1, v2] index lists.
    """
    tri  = Delaunay(pts)
    keep = []
    for simplex in tri.simplices:
        centroid = pts[simplex].mean(axis=0)
        if not point_in_polygon(centroid, hole_poly):
            keep.append(simplex.tolist())
    return pts, keep


def _find_boundary_edges(triangles: list, n_verts: int):
    """
    Return edges that appear in exactly one triangle (boundary edges).

    Each entry: (v_lo, v_hi) with v_lo < v_hi.
    """
    from collections import Counter
    edge_count: Counter = Counter()
    for tri in triangles:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            edge_count[tuple(sorted((a, b)))] += 1
    return [e for e, c in edge_count.items() if c == 1]


def _tag_boundary_edge(midpt: np.ndarray, hole_poly: np.ndarray) -> int:
    """
    Assign MFEM boundary attribute to an edge midpoint.
      1 = bottom (y ≈ 0)
      2 = right  (x ≈ 1)
      3 = top    (y ≈ 1)
      4 = left   (x ≈ 0)
      5 = hole
    """
    eps = 0.02
    x, y = midpt
    if y < eps:           return 1
    if x > 1.0 - eps:    return 2
    if y > 1.0 - eps:    return 3
    if x < eps:           return 4
    return 5              # hole boundary


# ---------------------------------------------------------------------------
# MFEM .mesh writer
# ---------------------------------------------------------------------------

def write_mfem_mesh(
    filepath: str,
    verts: np.ndarray,
    triangles: list,
    boundary_edges: list,
    edge_tags: list,
) -> None:
    """Write a 2-D triangle mesh in MFEM v1.0 text format."""
    lines = [
        "MFEM mesh v1.0",
        "",
        "dimension",
        "2",
        "",
        f"elements",
        f"{len(triangles)}",
    ]
    for tri in triangles:
        lines.append(f"1 2 {tri[0]} {tri[1]} {tri[2]}")

    lines += ["", "boundary", f"{len(boundary_edges)}"]
    for (v0, v1), tag in zip(boundary_edges, edge_tags):
        lines.append(f"{tag} 1 {v0} {v1}")

    lines += ["", "vertices", f"{len(verts)}", "2"]
    for v in verts:
        lines.append(f"{v[0]:.8f} {v[1]:.8f}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Top-level: single mesh
# ---------------------------------------------------------------------------

def create_fine_mesh(
    filename: str,
    center: tuple,
    base_radius: float,
    mesh_size_far:    float = 0.07,
    mesh_size_hole:   float = 0.025,
    refinement_depth: float = 0.08,
    seed: int = None,
) -> int:
    """
    Generate one plate-with-blob-hole fine mesh and write it as MFEM .mesh.

    Returns the number of elements in the final mesh.
    """
    # 1. Hole boundary
    hole_poly = generate_blob_points(center, base_radius, n_points=64, seed=seed)

    # 2. Point cloud
    outer_pts    = _make_boundary_pts(mesh_size_far)
    interior_pts = _make_interior_pts(
        hole_poly, mesh_size_far, mesh_size_hole, refinement_depth
    )
    all_pts = np.vstack([outer_pts, hole_poly, interior_pts])

    # Remove exact duplicates (can happen at corners)
    all_pts = np.unique(np.round(all_pts, 8), axis=0)

    # 3. Triangulate and cull
    verts, triangles = _triangulate_and_cull(all_pts, hole_poly)

    if not triangles:
        raise RuntimeError("No valid triangles after culling — check hole geometry.")

    # 4. Boundary edges and tags
    bdr_edges = _find_boundary_edges(triangles, len(verts))
    bdr_tags  = [
        _tag_boundary_edge((verts[e[0]] + verts[e[1]]) / 2, hole_poly)
        for e in bdr_edges
    ]

    # 5. Write
    write_mfem_mesh(filename, verts, triangles, bdr_edges, bdr_tags)
    return len(triangles)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_all(
    output_dir:              str,
    n_samples:               int   = 200,
    mesh_size_far:           float = 0.07,
    mesh_size_hole:          float = 0.025,
    refinement_depth:        float = 0.08,
    seed:                    int   = 42,
    start_index:             int   = 0,
    clear:                   bool  = True,
    near_boundary_fraction:  float = 0.0,
) -> None:
    """
    Generate n_samples fine plate-with-hole meshes in output_dir.

    Args:
        start_index:            First sample index (use >0 to append to existing dir).
        clear:                  Delete existing sample_*.mesh files before generating.
        near_boundary_fraction: Fraction of samples whose hole is placed close to
                                an edge (margin_factor=1.1 instead of 1.5).
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if clear:
        for f in Path(output_dir).glob("sample_*.mesh"):
            os.remove(f)

    mode_str = f"  near_boundary_fraction={near_boundary_fraction:.0%}" if near_boundary_fraction > 0 else ""
    print(f"Generating {n_samples} fine meshes → {output_dir}/  (start={start_index})")
    print(f"  mesh_size_far={mesh_size_far}  mesh_size_hole={mesh_size_hole}"
          f"  refinement_depth={refinement_depth}{mode_str}")

    n_elems_all = []
    for i in range(n_samples):
        base_radius = 0.12 + 0.10 * np.random.random()
        # Near-boundary: place hole much closer to an edge
        near = (np.random.random() < near_boundary_fraction)
        margin_factor = 1.1 if near else 1.5
        margin_offset = 0.02 if near else 0.10
        margin = base_radius * margin_factor + margin_offset
        cx = margin + (1 - 2 * margin) * np.random.random()
        cy = margin + (1 - 2 * margin) * np.random.random()

        idx      = start_index + i
        filename = os.path.join(output_dir, f"sample_{idx:03d}.mesh")
        try:
            n_e = create_fine_mesh(
                filename,
                center=(cx, cy),
                base_radius=base_radius,
                mesh_size_far=mesh_size_far,
                mesh_size_hole=mesh_size_hole,
                refinement_depth=refinement_depth,
                seed=2000 + idx,
            )
            n_elems_all.append(n_e)
            if (i + 1) % 20 == 0 or i == 0:
                nb_tag = " [near-bdr]" if near else ""
                print(f"  [{i+1:3d}/{n_samples}]  {Path(filename).name}  "
                      f"r={base_radius:.3f}  elems={n_e}{nb_tag}")
        except Exception as e:
            print(f"  [{i+1:3d}/{n_samples}]  {Path(filename).name}  FAILED: {e}")

    if n_elems_all:
        print(f"\nDone — {len(n_elems_all)}/{n_samples} meshes generated")
        print(f"  Elements: min={min(n_elems_all)}  "
              f"max={max(n_elems_all)}  "
              f"mean={int(np.mean(n_elems_all))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fine plate-with-hole triangle meshes (no gmsh required)"
    )
    parser.add_argument("--n",                       type=int,   default=200)
    parser.add_argument("--output",                  type=str,   default=None)
    parser.add_argument("--mesh-size-far",           type=float, default=0.07)
    parser.add_argument("--mesh-size-hole",          type=float, default=0.025)
    parser.add_argument("--refinement-depth",        type=float, default=0.08)
    parser.add_argument("--seed",                    type=int,   default=42)
    parser.add_argument("--start-index",             type=int,   default=0,
                        help="First sample index (for appending to existing dir)")
    parser.add_argument("--no-clear",                action="store_true",
                        help="Do not delete existing sample_*.mesh files")
    parser.add_argument("--near-boundary-fraction",  type=float, default=0.0,
                        help="Fraction of samples with hole near edge (0–1)")
    args = parser.parse_args()

    script_dir   = Path(__file__).parent
    project_root = script_dir.parent
    output_dir   = args.output or str(project_root / "train_fine")

    generate_all(
        output_dir             = output_dir,
        n_samples              = args.n,
        mesh_size_far          = args.mesh_size_far,
        mesh_size_hole         = args.mesh_size_hole,
        refinement_depth       = args.refinement_depth,
        seed                   = args.seed,
        start_index            = args.start_index,
        clear                  = not args.no_clear,
        near_boundary_fraction = args.near_boundary_fraction,
    )
