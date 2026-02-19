"""
Region-based morphing for Abaqus .inp meshes.

Reads a markdown config file that defines regions by geometric rules (no set names).
Assigns moving / anchor / morphing roles from geometry, then applies IDW morphing
with optional per-region p and dynamic anchor reassignment.

Morphing indices are managed internally via MorphingContext and can be exported
to VTU with PointData for visualization in ParaView (using PyVista).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# Morphing role constants
ROLE_MOVING = 0
ROLE_ANCHOR = 1
ROLE_MORPHING = 2

ROLE_NAMES = {
    ROLE_MOVING: "moving",
    ROLE_ANCHOR: "anchor",
    ROLE_MORPHING: "morphing",
}


@dataclass
class MorphingContext:
    """
    Internal morphing state for all nodes. Never written to INP file.

    Used to track node classifications dynamically based on delta_R,
    and can be exported to VTU with PointData for ParaView visualization.
    """
    # Node identification
    node_ids: np.ndarray              # (N,) original node IDs from INP
    coords_original: np.ndarray       # (N, 2/3) original coordinates
    coords_morphed: np.ndarray        # (N, 2/3) morphed coordinates

    # Morphing classification (computed based on delta_R)
    role: np.ndarray                  # (N,) 0=moving, 1=anchor, 2=morphing

    # Geometric data
    distance_from_center: np.ndarray  # (N,) radial distance from hole center
    displacement: np.ndarray          # (N, 2/3) computed displacement vectors

    # Configuration snapshot
    delta_R: float = 0.0              # Hole radius change applied
    R0: float = 0.0                   # Initial hole radius
    R_target: float = 0.0             # Target hole radius (R0 + delta_R)
    R_transition: float = 0.0         # Outer transition radius
    center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    def get_moving_mask(self) -> np.ndarray:
        """Return boolean mask for moving nodes."""
        return self.role == ROLE_MOVING

    def get_anchor_mask(self) -> np.ndarray:
        """Return boolean mask for anchor nodes."""
        return self.role == ROLE_ANCHOR

    def get_morphing_mask(self) -> np.ndarray:
        """Return boolean mask for morphing nodes."""
        return self.role == ROLE_MORPHING

    def get_role_name(self, node_idx: int) -> str:
        """Get human-readable role name for a node index."""
        return ROLE_NAMES.get(self.role[node_idx], "unknown")

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of morphing context."""
        return {
            "total_nodes": len(self.node_ids),
            "moving_count": int(np.sum(self.role == ROLE_MOVING)),
            "anchor_count": int(np.sum(self.role == ROLE_ANCHOR)),
            "morphing_count": int(np.sum(self.role == ROLE_MORPHING)),
            "delta_R": self.delta_R,
            "R0": self.R0,
            "R_target": self.R_target,
            "R_transition": self.R_transition,
            "max_displacement": float(np.max(np.linalg.norm(self.displacement, axis=1))),
        }

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
    delta_R: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign each node to a region and role using geometric rules.

    IMPORTANT: Classification is based on TARGET geometry (R0 + delta_R), not initial.
    This ensures nodes are correctly classified based on where they will be after morphing.

    Args:
        coords: (N, 2) or (N, 3) node coordinates.
        config: Loaded morphing config (with _R0, _R_transition, _tolerance, _hole_center, regions).
        delta_R: Hole radius change. Nodes are classified based on R_target = R0 + delta_R.

    Returns:
        region_ids: (N,) int array: 0 = hole_boundary, 1 = transition, 2 = far_field (order from config).
        roles: (N,) int array: 0 = moving, 1 = anchor, 2 = morphing.
        distances: (N,) float array: distance from each node to hole center.
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

    # Use TARGET radius for classification (R0 + delta_R)
    R_target = R0 + delta_R

    region_id = np.full(N, -1, dtype=int)

    # hole_boundary: nodes that will be ON the new hole boundary
    # These are nodes currently at distance ~ R0 (they will move to R_target)
    # But we also need to catch nodes that would be "swallowed" by the expanding hole
    if delta_R >= 0:
        # Expanding hole: original boundary nodes are moving nodes
        # Nodes at d in [R0 - tol, R0 + tol] are the original hole boundary
        mask_hole = (d >= R0 - tol) & (d <= R0 + tol)
    else:
        # Shrinking hole: nodes at current boundary move inward
        # Nodes at d in [R0 - tol, R0 + tol] are still the moving nodes
        mask_hole = (d >= R0 - tol) & (d <= R0 + tol)

    if "hole_boundary" in region_names:
        region_id[mask_hole] = region_names.index("hole_boundary")

    # transition: nodes between hole boundary and far field
    # For expanding hole: nodes in (R0 + tol, R_transition] that won't be swallowed
    # Adjust transition inner boundary based on delta_R
    transition_inner = R0 + tol
    if delta_R > 0:
        # When expanding, nodes closer than R_target + small margin become morphing
        # They need to move out of the way
        transition_inner = R0 + tol
    mask_trans = (d > transition_inner) & (d <= R_transition)
    if "transition" in region_names:
        region_id[mask_trans] = region_names.index("transition")

    # far_field: d > R_transition (always anchors)
    mask_far = d > R_transition
    if "far_field" in region_names:
        region_id[mask_far] = region_names.index("far_field")

    # Unassigned (e.g. d < R0 - tol, interior to hole): treat as far_field (anchor) if present
    unassigned = region_id < 0
    if unassigned.any() and "far_field" in region_names:
        region_id[unassigned] = region_names.index("far_field")

    # Map region -> role: moving=0, anchor=1, morphing=2
    role_map = {"moving": ROLE_MOVING, "anchor": ROLE_ANCHOR, "morphing": ROLE_MORPHING}
    roles = np.full(N, ROLE_ANCHOR, dtype=int)  # default anchor
    for rname, rdef in config.get("regions", {}).items():
        rid = region_names.index(rname) if rname in region_names else -1
        if rid < 0:
            continue
        role_str = rdef.get("role", "anchor")
        roles[region_id == rid] = role_map.get(role_str, ROLE_ANCHOR)

    return region_id, roles, d


def reassign_anchors_near_hole(
    coords: np.ndarray,
    region_id: np.ndarray,
    roles: np.ndarray,
    config: Dict[str, Any],
    region_names: List[str],
    delta_R: float = 0.0,
) -> np.ndarray:
    """
    Reassign anchor nodes that are too close to the TARGET hole boundary to morphing,
    to avoid "push past" and mesh collapse when hole grows.

    Uses R_target = R0 + delta_R for determining proximity.

    Modifies roles in place and returns the updated roles.
    """
    R0 = config["_R0"]
    R_target = R0 + delta_R  # Use target radius, not original
    min_dist = config["_min_anchor_distance"]
    center = config["_hole_center"]
    if center.size != coords.shape[1]:
        center = np.resize(center, coords.shape[1])
    d = distance_to_center(coords, center)

    # Distance from this node to the TARGET hole boundary (circle at R_target)
    dist_to_target_boundary = np.abs(d - R_target)

    # Also check distance to original boundary for nodes that might be swallowed
    dist_to_original_boundary = np.abs(d - R0)

    # Anchors that are within min_dist of either boundary -> morphing
    # This ensures nodes near the expanding hole path are properly handled
    near_target = dist_to_target_boundary < min_dist
    near_original = dist_to_original_boundary < min_dist

    # For expanding hole, also convert anchors that would be "inside" the new hole
    if delta_R > 0:
        would_be_inside = d < R_target + min_dist
        to_reassign = (roles == ROLE_ANCHOR) & (near_target | near_original | would_be_inside)
    else:
        to_reassign = (roles == ROLE_ANCHOR) & (near_target | near_original)

    roles[to_reassign] = ROLE_MORPHING
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
    return_context: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, MorphingContext]]:
    """
    Run region-based IDW morphing: load config, assign regions from geometry,
    optionally reassign anchors near the hole, compute displacements, update manager nodes.

    Does not write the .inp; caller can use AbaqusWriter for that.

    Args:
        manager: AbaqusManager with loaded .inp (nodes must be present).
        config_path: Path to the morphing .md config file.
        delta_R: Hole radius change (positive = enlarge hole).
        reassign_anchors: If True, reassign anchors too close to hole to morphing.
        return_context: If True, return MorphingContext along with coordinates.

    Returns:
        If return_context=False:
            new_coords: (N, 2) or (N, 3) updated coordinates.
        If return_context=True:
            (new_coords, MorphingContext) tuple.
    """
    if manager.nodes is None:
        raise ValueError("Manager has no nodes")

    config = load_morphing_config(config_path)
    coords = manager.nodes.data.copy()
    n_nodes = coords.shape[0]
    region_names = list(config.get("regions", {}).keys())

    # Pass delta_R for dynamic classification based on target geometry
    region_id, roles, distances = assign_regions_from_geometry(coords, config, delta_R)
    if reassign_anchors:
        roles = reassign_anchors_near_hole(coords, region_id, roles, config, region_names, delta_R)

    moving_mask = roles == ROLE_MOVING
    anchor_mask = roles == ROLE_ANCHOR
    morphing_mask = roles == ROLE_MORPHING

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

    if return_context:
        ctx = MorphingContext(
            node_ids=manager.nodes.ids.copy(),
            coords_original=coords,
            coords_morphed=new_coords,
            role=roles,
            distance_from_center=distances,
            displacement=disp,
            delta_R=delta_R,
            R0=config["_R0"],
            R_target=config["_R0"] + delta_R,
            R_transition=config["_R_transition"],
            center=center,
        )
        return new_coords, ctx

    return new_coords


def morph_and_write(
    inp_path: Union[str, Path],
    config_path: Union[str, Path],
    output_path: Union[str, Path],
    delta_R: float,
    reassign_anchors: bool = True,
    export_debug_vtu: bool = False,
) -> Union[Tuple[str, str], Tuple[str, str, str]]:
    """
    Load .inp, run morphing from .md config, write morphed .inp and .vtu to output_path (same stem).

    Args:
        inp_path: Input .inp file.
        config_path: Morphing config .md file.
        output_path: Output .inp file path (e.g. outputs/OutputInp2D_morphed.inp); .vtu is written with same stem.
        delta_R: Hole radius change.
        reassign_anchors: Whether to reassign anchors near hole to morphing.
        export_debug_vtu: If True, also export a debug VTU with morphing indices for ParaView.

    Returns:
        If export_debug_vtu=False:
            (inp_path, vtu_path) paths to the written files.
        If export_debug_vtu=True:
            (inp_path, vtu_path, debug_vtu_path) paths to the written files.
    """
    manager = AbaqusManager(str(inp_path))
    result = run_morphing(
        manager, config_path, delta_R,
        reassign_anchors=reassign_anchors,
        return_context=export_debug_vtu
    )

    if export_debug_vtu:
        new_coords, ctx = result
    else:
        new_coords = result
        ctx = None

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inp_out, vtu_out = write_inp_and_vtu(manager, str(out_path))

    if export_debug_vtu and ctx is not None:
        debug_vtu_path = out_path.with_stem(out_path.stem + "_debug").with_suffix(".vtu")
        export_morphing_context_to_vtu(manager, ctx, str(debug_vtu_path))
        return inp_out, vtu_out, str(debug_vtu_path)

    return inp_out, vtu_out


def export_morphing_context_to_vtu(
    manager: AbaqusManager,
    ctx: MorphingContext,
    vtu_path: Union[str, Path],
    show_preview: bool = False,
) -> str:
    """
    Export morphing context to VTU with PointData for ParaView visualization.

    Uses PyVista to create an UnstructuredGrid with morphing indices as point data.
    This allows visualization and inspection of node classifications in ParaView.

    Args:
        manager: AbaqusManager with elements data.
        ctx: MorphingContext with morphing state.
        vtu_path: Output path for the debug VTU file.
        show_preview: If True, display interactive preview (requires display).

    Returns:
        Path to the written VTU file.

    PointData fields written:
        - MorphingRole: 0=moving, 1=anchor, 2=morphing
        - NodeID: Original node ID from INP file
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "PyVista is required for debug VTU export. "
            "Install with: pip install pyvista"
        )

    if manager.elements is None:
        raise ValueError("Manager has no elements")

    path = Path(vtu_path)

    # Build points array (ensure 3D for VTK)
    points = ctx.coords_morphed.copy()
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])

    # Build node ID to index mapping
    node_id_to_idx = {int(nid): i for i, nid in enumerate(ctx.node_ids)}

    # Build cells for PyVista
    # PyVista cell format: [n_points, p0, p1, ..., pn, n_points, p0, ...]
    cells = []
    cell_types = []

    # VTK cell type constants
    VTK_TRIANGLE = 5
    VTK_QUAD = 9

    for i in range(len(manager.elements.ids)):
        conn = manager.elements.data[i]
        n_nodes_elem = len(conn)

        # Convert node IDs to 0-based indices
        indices = [node_id_to_idx[int(nid)] for nid in conn]

        # Add cell: [n_nodes, idx0, idx1, ...]
        cells.append(n_nodes_elem)
        cells.extend(indices)

        # Determine cell type
        if n_nodes_elem == 3:
            cell_types.append(VTK_TRIANGLE)
        elif n_nodes_elem == 4:
            cell_types.append(VTK_QUAD)
        elif n_nodes_elem == 6:
            cell_types.append(VTK_TRIANGLE)  # Quadratic triangle
        elif n_nodes_elem == 8:
            cell_types.append(VTK_QUAD)  # Quadratic quad
        else:
            cell_types.append(VTK_QUAD)  # Default

    cells = np.array(cells)
    cell_types = np.array(cell_types)

    # Create PyVista UnstructuredGrid
    mesh = pv.UnstructuredGrid(cells, cell_types, points)

    # Add PointData fields for ParaView visualization
    mesh.point_data["MorphingRole"] = ctx.role
    mesh.point_data["NodeID"] = ctx.node_ids

    # Save to VTU
    mesh.save(str(path))

    # Optional preview
    if show_preview:
        try:
            plotter = pv.Plotter()
            plotter.add_mesh(
                mesh,
                scalars="MorphingRole",
                cmap="coolwarm",
                show_edges=True,
                scalar_bar_args={"title": "Role: 0=moving, 1=anchor, 2=morphing"}
            )
            plotter.add_title(f"Morphing Context (delta_R={ctx.delta_R:.3f})")
            plotter.show()
        except Exception as e:
            print(f"Preview not available: {e}")

    return str(path)


def preview_morphing_context(
    manager: AbaqusManager,
    ctx: MorphingContext,
    scalars: str = "MorphingRole",
) -> None:
    """
    Display interactive preview of morphing context using PyVista.

    Args:
        manager: AbaqusManager with elements data.
        ctx: MorphingContext with morphing state.
        scalars: Field to color by ("MorphingRole", "RegionID", "DistanceFromCenter", etc.)
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".vtu", delete=False) as f:
        vtu_path = f.name

    export_morphing_context_to_vtu(manager, ctx, vtu_path, show_preview=False)

    try:
        import pyvista as pv
        mesh = pv.read(vtu_path)

        # Define colormaps based on scalar type
        if scalars == "MorphingRole":
            cmap = "coolwarm"
            title = "Role: 0=moving, 1=anchor, 2=morphing"
        else:
            cmap = "plasma"
            title = scalars

        plotter = pv.Plotter()
        plotter.add_mesh(
            mesh,
            scalars=scalars,
            cmap=cmap,
            show_edges=True,
            scalar_bar_args={"title": title}
        )
        plotter.add_title(f"delta_R={ctx.delta_R:.3f}, R_target={ctx.R_target:.3f}")
        plotter.show()
    finally:
        Path(vtu_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python morphing.py <input.inp> <config.md> <delta_R> [output.inp] [options]"
        )
        print("  If output.inp is omitted, writes outputs/OutputInp2D_morphed.inp and .vtu")
        print("")
        print("Options:")
        print("  --no-reassign   Disable dynamic anchor reassignment")
        print("  --debug         Export debug VTU with morphing indices for ParaView")
        print("  --preview       Show interactive PyVista preview (requires display)")
        print("")
        print("Example:")
        print("  python morphing.py inputs/BaseInp2D.inp configs/quarter_plate_with_hole_morphing.md 0.5 --debug")
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
    debug_mode = "--debug" in sys.argv
    preview_mode = "--preview" in sys.argv

    result = morph_and_write(
        inp_path, config_path, output_path, delta_R,
        reassign_anchors=reassign,
        export_debug_vtu=debug_mode
    )

    if debug_mode:
        inp_out, vtu_out, debug_vtu_out = result
        print(f"Morphed .inp written to {inp_out}")
        print(f"Morphed .vtu written to {vtu_out}")
        print(f"Debug .vtu written to {debug_vtu_out}")
        print("")
        print("Open the debug VTU in ParaView and color by 'MorphingRole':")
        print("  0 = moving (hole boundary)")
        print("  1 = anchor (far field)")
        print("  2 = morphing (transition)")
    else:
        inp_out, vtu_out = result
        print(f"Morphed .inp written to {inp_out}")
        print(f"Morphed .vtu written to {vtu_out}")

    # Optional preview
    if preview_mode:
        print("\nLaunching PyVista preview...")
        manager = AbaqusManager(str(inp_path))
        _, ctx = run_morphing(
            manager, config_path, delta_R,
            reassign_anchors=reassign,
            return_context=True
        )
        preview_morphing_context(manager, ctx, scalars="MorphingRole")
