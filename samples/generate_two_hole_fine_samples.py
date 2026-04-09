"""
Generate fine mesh samples with TWO blob holes using scipy Delaunay.

No gmsh required.  Writes MFEM .mesh v1.0 directly.

Both holes receive boundary tag 5 (traction-free by default in the solver).
Outer boundary tags: bottom=1, right=2, top=3, left=4.

Usage:
    # Append 100 near-boundary double-hole meshes starting at index 200
    python3 samples/generate_two_hole_fine_samples.py \\
        --n 100 --output train02 --start-index 200 --no-clear \\
        --near-boundary-fraction 1.0
"""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# Shared geometry helpers  (duplicated from generate_fine_samples.py)
# ---------------------------------------------------------------------------

def generate_blob_points(center, base_radius, n_points=48, seed=None):
    if seed is not None:
        np.random.seed(seed)
    cx, cy = center
    available_modes = [2, 3, 4, 5, 6, 7]
    n_modes = np.random.randint(2, 5)
    selected_modes = np.random.choice(available_modes, size=n_modes, replace=False)
    amplitudes, phases = [], []
    for k in selected_modes:
        amplitudes.append(base_radius * np.random.uniform(0.08, 0.35) / np.sqrt(k))
        phases.append(np.random.uniform(0, 2 * np.pi))
    aspect_ratio     = np.random.uniform(0.6, 1.4)
    elongation_angle = np.random.uniform(0, 2 * np.pi)
    n_bumps          = np.random.randint(0, 4)
    bump_positions   = np.random.uniform(0, 2 * np.pi, n_bumps)
    bump_widths      = np.random.uniform(0.3, 0.8, n_bumps)
    bump_heights     = np.random.uniform(-0.15, 0.2, n_bumps) * base_radius
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


def point_in_polygon(pt, poly):
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


def dist_to_polygon_boundary(pt, poly):
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


def _make_boundary_pts(mesh_size):
    n   = max(3, int(round(1.0 / mesh_size)))
    ts  = np.linspace(0, 1, n + 1)
    pts = []
    for t in ts:
        pts.extend([[t, 0], [t, 1], [0, t], [1, t]])
    return np.unique(np.array(pts), axis=0)


def write_mfem_mesh(filepath, verts, triangles, boundary_edges, edge_tags):
    lines = ["MFEM mesh v1.0", "", "dimension", "2", "",
             f"elements", f"{len(triangles)}"]
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
# Multi-hole mesh generation
# ---------------------------------------------------------------------------

def _make_interior_pts_multi(hole_polys, mesh_size_far, mesh_size_hole,
                              refinement_depth, jitter=0.3):
    """Interior points with local refinement near ALL hole boundaries."""
    rng = np.random.default_rng(0)
    safe_lo = 0.5 * mesh_size_far
    safe_hi = 1.0 - safe_lo

    def _jitter_clamp(x, y, h):
        jx = rng.uniform(-jitter * h, jitter * h)
        jy = rng.uniform(-jitter * h, jitter * h)
        return [float(np.clip(x + jx, safe_lo, safe_hi)),
                float(np.clip(y + jy, safe_lo, safe_hi))]

    # Combined bounding box covering all refinement zones
    all_bb_lo = np.array([1.0, 1.0])
    all_bb_hi = np.array([0.0, 0.0])
    for poly in hole_polys:
        bb_lo = poly.min(axis=0) - refinement_depth - 0.01
        bb_hi = poly.max(axis=0) + refinement_depth + 0.01
        all_bb_lo = np.minimum(all_bb_lo, np.maximum(bb_lo, 0.005))
        all_bb_hi = np.maximum(all_bb_hi, np.minimum(bb_hi, 0.995))

    pts = []

    # Fine grid near any hole
    xs = np.arange(all_bb_lo[0], all_bb_hi[0] + mesh_size_hole * 0.5, mesh_size_hole)
    ys = np.arange(all_bb_lo[1], all_bb_hi[1] + mesh_size_hole * 0.5, mesh_size_hole)
    for x in xs:
        for y in ys:
            p = np.array([x, y])
            # Skip if inside any hole
            if any(point_in_polygon(p, poly) for poly in hole_polys):
                continue
            # Include if near any hole
            if any(dist_to_polygon_boundary(p, poly) <= refinement_depth
                   for poly in hole_polys):
                pts.append(_jitter_clamp(x, y, mesh_size_hole))

    # Coarse grid everywhere else
    margin = mesh_size_far * 0.5
    xs_c = np.arange(margin, 1.0, mesh_size_far)
    ys_c = np.arange(margin, 1.0, mesh_size_far)
    for x in xs_c:
        for y in ys_c:
            p = np.array([x, y])
            if any(point_in_polygon(p, poly) for poly in hole_polys):
                continue
            if any(dist_to_polygon_boundary(p, poly) <= refinement_depth
                   for poly in hole_polys):
                continue
            pts.append(_jitter_clamp(x, y, mesh_size_far))

    return np.array(pts) if pts else np.empty((0, 2))


def _triangulate_and_cull_multi(pts, hole_polys):
    tri  = Delaunay(pts)
    keep = []
    for simplex in tri.simplices:
        centroid = pts[simplex].mean(axis=0)
        if not any(point_in_polygon(centroid, poly) for poly in hole_polys):
            keep.append(simplex.tolist())
    return pts, keep


def _find_boundary_edges(triangles):
    from collections import Counter
    edge_count = Counter()
    for tri in triangles:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            edge_count[tuple(sorted((a, b)))] += 1
    return [e for e, c in edge_count.items() if c == 1]


def _tag_boundary_edge(midpt, hole_polys):
    eps = 0.02
    x, y = midpt
    if y < eps:           return 1
    if x > 1.0 - eps:    return 2
    if y > 1.0 - eps:    return 3
    if x < eps:           return 4
    return 5   # any hole boundary


# ---------------------------------------------------------------------------
# Single mesh: two holes
# ---------------------------------------------------------------------------

def _sample_hole(base_radius_range, margin_factor, margin_offset, existing_holes,
                 min_gap=0.05, max_attempts=200, seed_offset=0):
    """Sample a (center, radius) for a new hole that doesn't overlap existing ones."""
    for attempt in range(max_attempts):
        r  = base_radius_range[0] + (base_radius_range[1] - base_radius_range[0]) * np.random.random()
        mg = r * margin_factor + margin_offset
        cx = mg + (1 - 2 * mg) * np.random.random()
        cy = mg + (1 - 2 * mg) * np.random.random()
        # Check overlap with all existing holes
        ok = True
        for (ec, er) in existing_holes:
            if np.linalg.norm(np.array([cx, cy]) - np.array(ec)) < r + er + min_gap:
                ok = False
                break
        if ok:
            return (cx, cy), r
    return None, None  # placement failed


def create_two_hole_fine_mesh(filename, margin_factor=1.4, margin_offset=0.08,
                               mesh_size_far=0.07, mesh_size_hole=0.025,
                               refinement_depth=0.08, seed=None):
    """Generate one plate with two non-overlapping blob holes."""
    if seed is not None:
        np.random.seed(seed)

    base_radius_range = (0.10, 0.18)
    holes = []  # list of ((cx,cy), r)

    for h in range(2):
        c, r = _sample_hole(base_radius_range, margin_factor, margin_offset, holes)
        if c is None:
            raise RuntimeError(f"Could not place hole {h+1} without overlap after many attempts")
        holes.append((c, r))

    hole_polys = [
        generate_blob_points(c, r, n_points=64, seed=seed + h if seed else None)
        for h, (c, r) in enumerate(holes)
    ]

    outer_pts    = _make_boundary_pts(mesh_size_far)
    interior_pts = _make_interior_pts_multi(hole_polys, mesh_size_far,
                                            mesh_size_hole, refinement_depth)
    all_pts = np.vstack([outer_pts] + hole_polys + [interior_pts])
    all_pts = np.unique(np.round(all_pts, 8), axis=0)

    verts, triangles = _triangulate_and_cull_multi(all_pts, hole_polys)
    if not triangles:
        raise RuntimeError("No valid triangles after culling")

    bdr_edges = _find_boundary_edges(triangles)
    bdr_tags  = [
        _tag_boundary_edge((verts[e[0]] + verts[e[1]]) / 2, hole_polys)
        for e in bdr_edges
    ]
    write_mfem_mesh(filename, verts, triangles, bdr_edges, bdr_tags)
    return len(triangles)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_all(output_dir, n_samples=200, mesh_size_far=0.07,
                 mesh_size_hole=0.025, refinement_depth=0.08, seed=42,
                 start_index=0, clear=True, near_boundary_fraction=0.0):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    if clear:
        for f in Path(output_dir).glob("sample_*.mesh"):
            os.remove(f)

    mode_str = f"  near_boundary_fraction={near_boundary_fraction:.0%}" if near_boundary_fraction > 0 else ""
    print(f"Generating {n_samples} two-hole fine meshes → {output_dir}/  (start={start_index})")
    print(f"  mesh_size_far={mesh_size_far}  mesh_size_hole={mesh_size_hole}"
          f"  refinement_depth={refinement_depth}{mode_str}")

    n_elems_all = []
    for i in range(n_samples):
        near          = (np.random.random() < near_boundary_fraction)
        margin_factor = 1.1 if near else 1.4
        margin_offset = 0.02 if near else 0.08
        idx      = start_index + i
        filename = os.path.join(output_dir, f"sample_{idx:03d}.mesh")
        try:
            n_e = create_two_hole_fine_mesh(
                filename,
                margin_factor=margin_factor,
                margin_offset=margin_offset,
                mesh_size_far=mesh_size_far,
                mesh_size_hole=mesh_size_hole,
                refinement_depth=refinement_depth,
                seed=3000 + idx,
            )
            n_elems_all.append(n_e)
            if (i + 1) % 20 == 0 or i == 0:
                nb_tag = " [near-bdr]" if near else ""
                print(f"  [{i+1:3d}/{n_samples}]  {Path(filename).name}  elems={n_e}{nb_tag}")
        except Exception as e:
            print(f"  [{i+1:3d}/{n_samples}]  {Path(filename).name}  FAILED: {e}")

    if n_elems_all:
        print(f"\nDone — {len(n_elems_all)}/{n_samples} meshes generated")
        print(f"  Elements: min={min(n_elems_all)}  max={max(n_elems_all)}  "
              f"mean={int(np.mean(n_elems_all))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fine two-hole plate meshes (no gmsh required)"
    )
    parser.add_argument("--n",                       type=int,   default=200)
    parser.add_argument("--output",                  type=str,   default=None)
    parser.add_argument("--mesh-size-far",           type=float, default=0.07)
    parser.add_argument("--mesh-size-hole",          type=float, default=0.025)
    parser.add_argument("--refinement-depth",        type=float, default=0.08)
    parser.add_argument("--seed",                    type=int,   default=42)
    parser.add_argument("--start-index",             type=int,   default=0)
    parser.add_argument("--no-clear",                action="store_true")
    parser.add_argument("--near-boundary-fraction",  type=float, default=0.0)
    args = parser.parse_args()

    script_dir   = Path(__file__).parent
    project_root = script_dir.parent
    output_dir   = args.output or str(project_root / "train02")

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
