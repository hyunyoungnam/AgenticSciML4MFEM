"""
Region-based morphing for Abaqus .inp meshes.

Reads a markdown config file that defines regions by geometric rules (no set names).
Assigns moving / anchor / morphing roles from geometry, then applies IDW morphing
with optional per-region p and dynamic anchor reassignment.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def _parse_yaml_minimal(yaml_str: str) -> Dict[str, Any]:
    """
    Minimal parser for our morphing config YAML (no PyYAML dependency).
    Extracts geometry, reassignment, and regions (role, idw_p) via regex.
    """
    config: Dict[str, Any] = {}
    geo: Dict[str, Any] = {}
    # hole_center: [0.0, 0.0]
    m = re.search(r"hole_center:\s*\[([^\]]+)\]", yaml_str)
    if m:
        geo["hole_center"] = [float(x.strip()) for x in m.group(1).split(",")]
    else:
        geo["hole_center"] = [0.0, 0.0]
    for key, default in (
        ("initial_hole_radius", 2.5),
        ("transition_outer_radius", 8.0),
        ("tolerance", 0.15),
    ):
        m = re.search(rf"{key}:\s*([\d.]+)", yaml_str)
        geo[key] = float(m.group(1)) if m else default
    config["geometry"] = geo

    reassign: Dict[str, Any] = {}
    m = re.search(r"min_anchor_distance_from_hole:\s*([\d.]+)", yaml_str)
    reassign["min_anchor_distance_from_hole"] = float(m.group(1)) if m else 0.5
    config["reassignment"] = reassign

    regions: Dict[str, Any] = {}
    for rname in ("hole_boundary", "transition", "far_field"):
        # Find block "  rname:" then "role: ..." and "idw_p: ..."
        pat = rf"{rname}:\s*\n(.*?)(?=\n  \w|\n\n|\Z)"
        m = re.search(pat, yaml_str, re.DOTALL)
        if not m:
            continue
        block = m.group(1)
        role_m = re.search(r"role:\s*(\w+)", block)
        idw_m = re.search(r"idw_p:\s*([\d.]+|null)", block)
        regions[rname] = {
            "role": role_m.group(1) if role_m else "anchor",
            "idw_p": float(idw_m.group(1)) if idw_m and idw_m.group(1) != "null" else None,
        }
    config["regions"] = regions
    return config
from manager import AbaqusManager
from writer import AbaqusWriter, OUTPUT_MORPHED_INP, write_inp_and_vtu


def load_morphing_config(md_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load morphing configuration from a markdown file.
    Expects a single YAML code block (```yaml ... ```). Resolves symbol references
    (R0, R_transition, tolerance, delta_R) from the geometry and design_variable sections.

    Args:
        md_path: Path to the .md file.

    Returns:
        Parsed config dict with geometry.initial_hole_radius, geometry.transition_outer_radius,
        geometry.tolerance, regions, reassignment, design_variable, etc.
    """
    path = Path(md_path)
    if not path.exists():
        raise FileNotFoundError(f"Morphing config not found: {path}")

    text = path.read_text(encoding="utf-8")
    block = re.search(r"```yaml\s*(.*?)\s*```", text, re.DOTALL)
    if not block:
        raise ValueError(f"No YAML block found in {path}")

    yaml_str = block.group(1).strip()
    if yaml is not None:
        config = yaml.safe_load(yaml_str)
    else:
        config = _parse_yaml_minimal(yaml_str)
    if not config:
        raise ValueError("YAML block is empty")

    # Resolve numeric geometry for convenience
    geo = config.get("geometry", {})
    config["_R0"] = float(geo.get("initial_hole_radius", 0.0))
    config["_R_transition"] = float(geo.get("transition_outer_radius", 1.0))
    config["_tolerance"] = float(geo.get("tolerance", 0.1))
    config["_hole_center"] = np.array(geo.get("hole_center", [0.0, 0.0]), dtype=float)

    reassign = config.get("reassignment", {})
    config["_min_anchor_distance"] = float(reassign.get("min_anchor_distance_from_hole", 0.5))

    # Parse symmetry constraints
    symmetry = config.get("symmetry", {})
    config["_symmetry"] = []
    for key, val in symmetry.items():
        if isinstance(val, dict) and "axis" in val:
            config["_symmetry"].append({
                "axis": int(val["axis"]),
                "tolerance": float(val.get("tolerance", 1e-6)),
            })

    return config


def distance_to_center(coords: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Return (N,) distances of each row of coords to center. 2D or 3D."""
    d = coords - np.atleast_1d(center)[: coords.shape[1]]
    return np.sqrt(np.sum(d * d, axis=1))


def assign_regions_from_geometry(
    coords: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each node to a region and role using geometric rules only.

    Args:
        coords: (N, 2) or (N, 3) node coordinates.
        config: Loaded morphing config (with _R0, _R_transition, _tolerance, _hole_center, regions).

    Returns:
        region_ids: (N,) int array: 0 = hole_boundary, 1 = transition, 2 = far_field (order from config).
        roles: (N,) int array: 0 = moving, 1 = anchor, 2 = morphing.
    """
    R0 = config["_R0"]
    R_transition = config["_R_transition"]
    tol = config["_tolerance"]
    center = config["_hole_center"]
    if center.size != coords.shape[1]:
        center = np.resize(center, coords.shape[1])

    d = distance_to_center(coords, center)
    N = coords.shape[0]
    region_names = list(config.get("regions", {}).keys())
    if not region_names:
        raise ValueError("Config has no regions")

    region_id = np.full(N, -1, dtype=int)
    # hole_boundary: d in [R0 - tol, R0 + tol]
    mask_hole = (d >= R0 - tol) & (d <= R0 + tol)
    if "hole_boundary" in region_names:
        region_id[mask_hole] = region_names.index("hole_boundary")
    # transition: d in (R0 + tol, R_transition]
    mask_trans = (d > R0 + tol) & (d <= R_transition)
    if "transition" in region_names:
        region_id[mask_trans] = region_names.index("transition")
    # far_field: d > R_transition
    mask_far = d > R_transition
    if "far_field" in region_names:
        region_id[mask_far] = region_names.index("far_field")

    # Unassigned (e.g. d < R0 - tol, interior to hole): treat as far_field (anchor) if present
    unassigned = region_id < 0
    if unassigned.any() and "far_field" in region_names:
        region_id[unassigned] = region_names.index("far_field")

    # Map region -> role: moving=0, anchor=1, morphing=2
    role_map = {"moving": 0, "anchor": 1, "morphing": 2}
    roles = np.full(N, 1, dtype=int)  # default anchor
    for rname, rdef in config.get("regions", {}).items():
        rid = region_names.index(rname) if rname in region_names else -1
        if rid < 0:
            continue
        role_str = rdef.get("role", "anchor")
        roles[region_id == rid] = role_map.get(role_str, 1)

    return region_id, roles


def reassign_anchors_near_hole(
    coords: np.ndarray,
    region_id: np.ndarray,
    roles: np.ndarray,
    config: Dict[str, Any],
    region_names: List[str],
) -> np.ndarray:
    """
    Reassign anchor nodes that are too close to the hole boundary to morphing,
    to avoid "push past" and mesh collapse when hole grows.

    Modifies roles in place and returns the updated roles.
    """
    R0 = config["_R0"]
    min_dist = config["_min_anchor_distance"]
    center = config["_hole_center"]
    if center.size != coords.shape[1]:
        center = np.resize(center, coords.shape[1])
    d = distance_to_center(coords, center)
    # Distance from this node to the hole "boundary" (circle at R0)
    dist_to_hole_boundary = np.abs(d - R0)
    anchor = 1
    morphing = 2
    # Anchors that are within min_dist of the hole boundary -> morphing
    to_reassign = (roles == anchor) & (dist_to_hole_boundary < min_dist)
    roles[to_reassign] = morphing
    return roles


def compute_moving_displacements(
    coords: np.ndarray,
    moving_mask: np.ndarray,
    center: np.ndarray,
    delta_R: float,
) -> np.ndarray:
    """
    Compute displacement vectors for moving nodes: radial outward by delta_R.
    Unit radial vector from center to node, then scale by delta_R.

    Returns:
        disp: (N, 2) or (N, 3); zero for non-moving nodes.
    """
    disp = np.zeros_like(coords)
    if not np.any(moving_mask):
        return disp
    c = np.resize(center, coords.shape[1])
    d = coords - c
    dist = np.sqrt(np.sum(d * d, axis=1))
    dist = np.where(dist < 1e-12, 1e-12, dist)
    radial = d / dist[:, np.newaxis]
    disp[moving_mask] = radial[moving_mask] * delta_R
    return disp


def apply_symmetry_constraints(
    coords: np.ndarray,
    disp: np.ndarray,
    symmetry_constraints: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Apply symmetry constraints to displacements.

    For nodes on a symmetry axis (e.g., x=0), zero out the displacement
    component perpendicular to that axis (e.g., dx=0 for x=0 symmetry).

    Args:
        coords: (N, 2) or (N, 3) node coordinates.
        disp: (N, 2) or (N, 3) displacement vectors.
        symmetry_constraints: List of {"axis": int, "tolerance": float}.
            axis=0 means x=0 symmetry (nodes with |x| < tol have dx=0)
            axis=1 means y=0 symmetry (nodes with |y| < tol have dy=0)

    Returns:
        Modified displacement array with symmetry enforced.
    """
    disp = disp.copy()
    for constraint in symmetry_constraints:
        axis = constraint["axis"]
        tol = constraint["tolerance"]
        if axis < coords.shape[1]:
            # Find nodes on this symmetry axis
            on_axis = np.abs(coords[:, axis]) < tol
            # Zero out displacement in the axis direction for these nodes
            disp[on_axis, axis] = 0.0
    return disp


def idw_displacements(
    coords: np.ndarray,
    moving_mask: np.ndarray,
    anchor_mask: np.ndarray,
    morphing_mask: np.ndarray,
    moving_disp: np.ndarray,
    p_per_node: np.ndarray,
    p_default: float = 2.0,
) -> np.ndarray:
    """
    Compute displacements for morphing nodes using Inverse Distance Weighting.
    disp_i = sum_j ( w_ij * disp_j ) / sum_j w_ij, where j runs over moving and anchor nodes,
    w_ij = 1 / dist_ij^p_i, and anchor disp_j = 0.

    So only moving nodes contribute non-zero disp_j; anchors contribute to the weight sum only.

    Args:
        coords: (N, 2) or (N, 3).
        moving_mask, anchor_mask, morphing_mask: boolean (N,).
        moving_disp: (N, 2) or (N, 3); only moving entries are non-zero.
        p_per_node: (N,) p-value for each morphing node (use p_default for others).
        p_default: used where p_per_node is invalid or zero.

    Returns:
        disp: (N, 2) or (N, 3); moving and anchor already set (moving_disp and 0); morphing filled.
    """
    disp = moving_disp.copy()
    if not np.any(morphing_mask):
        return disp

    N = coords.shape[0]
    source = moving_mask | anchor_mask
    if not np.any(source):
        return disp

    morph_idx = np.where(morphing_mask)[0]
    src_idx = np.where(source)[0]

    for i in morph_idx:
        p = p_per_node[i] if p_per_node[i] > 0 else p_default
        diff = coords[src_idx] - coords[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        dist = np.where(dist < 1e-12, 1e-12, dist)
        w = 1.0 / (dist ** p)
        d_j = disp[src_idx]
        numer = np.sum(w[:, np.newaxis] * d_j, axis=0)
        denom = np.sum(w)
        if denom > 1e-20:
            disp[i] = numer / denom
        else:
            disp[i] = 0.0

    return disp


def run_morphing(
    manager: AbaqusManager,
    config_path: Union[str, Path],
    delta_R: float,
    reassign_anchors: bool = True,
) -> np.ndarray:
    """
    Run region-based IDW morphing: load config, assign regions from geometry,
    optionally reassign anchors near the hole, compute displacements, update manager nodes.

    Does not write the .inp; caller can use AbaqusWriter for that.

    Args:
        manager: AbaqusManager with loaded .inp (nodes must be present).
        config_path: Path to the morphing .md config file.
        delta_R: Hole radius change (positive = enlarge hole).
        reassign_anchors: If True, reassign anchors too close to hole to morphing.

    Returns:
        new_coords: (N, 2) or (N, 3) updated coordinates (same order as manager.nodes.ids).
    """
    if manager.nodes is None:
        raise ValueError("Manager has no nodes")

    config = load_morphing_config(config_path)
    coords = manager.nodes.data.copy()
    n_nodes = coords.shape[0]
    region_names = list(config.get("regions", {}).keys())

    region_id, roles = assign_regions_from_geometry(coords, config)
    if reassign_anchors:
        roles = reassign_anchors_near_hole(coords, region_id, roles, config, region_names)

    moving_mask = roles == 0
    anchor_mask = roles == 1
    morphing_mask = roles == 2

    center = config["_hole_center"]
    if center.size != coords.shape[1]:
        center = np.resize(center, coords.shape[1])

    moving_disp = compute_moving_displacements(coords, moving_mask, center, delta_R)

    # Per-node p for IDW: from region's idw_p (morphing nodes only)
    p_per_node = np.zeros(n_nodes)
    for rname, rdef in config.get("regions", {}).items():
        rid = region_names.index(rname) if rname in region_names else -1
        if rid < 0:
            continue
        p_val = rdef.get("idw_p")
        if p_val is None:
            p_val = 2.0
        p_per_node[region_id == rid] = float(p_val)

    disp = idw_displacements(
        coords,
        moving_mask,
        anchor_mask,
        morphing_mask,
        moving_disp,
        p_per_node,
        p_default=2.0,
    )

    # Apply symmetry constraints (e.g., for quarter-plate models)
    symmetry_constraints = config.get("_symmetry", [])
    if symmetry_constraints:
        disp = apply_symmetry_constraints(coords, disp, symmetry_constraints)

    new_coords = coords + disp
    manager.update_nodes(new_coords)
    return new_coords


def morph_and_write(
    inp_path: Union[str, Path],
    config_path: Union[str, Path],
    output_path: Union[str, Path],
    delta_R: float,
    reassign_anchors: bool = True,
) -> Tuple[str, str]:
    """
    Load .inp, run morphing from .md config, write morphed .inp and .vtu to output_path (same stem).

    Args:
        inp_path: Input .inp file.
        config_path: Morphing config .md file.
        output_path: Output .inp file path (e.g. outputs/OutputInp2D_morphed.inp); .vtu is written with same stem.
        delta_R: Hole radius change.
        reassign_anchors: Whether to reassign anchors near hole to morphing.

    Returns:
        (inp_path, vtu_path) paths to the written files.
    """
    manager = AbaqusManager(str(inp_path))
    run_morphing(manager, config_path, delta_R, reassign_anchors=reassign_anchors)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inp_out, vtu_out = write_inp_and_vtu(manager, str(out_path))
    return inp_out, vtu_out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python morphing.py <input.inp> <config.md> <delta_R> [output.inp] [--no-reassign]"
        )
        print("  If output.inp is omitted, writes outputs/OutputInp2D_morphed.inp and .vtu")
        sys.exit(1)

    inp_path = sys.argv[1]
    config_path = sys.argv[2]
    delta_R = float(sys.argv[3])
    # Optional 4th arg: output path; else use standard name
    if len(sys.argv) >= 5 and not sys.argv[4].strip().startswith("--"):
        output_path = sys.argv[4]
    else:
        out_dir = Path(__file__).parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / OUTPUT_MORPHED_INP)
    reassign = "--no-reassign" not in sys.argv

    inp_out, vtu_out = morph_and_write(
        inp_path, config_path, output_path, delta_R, reassign_anchors=reassign
    )
    print(f"Morphed .inp written to {inp_out}")
    print(f"Morphed .vtu written to {vtu_out}")
