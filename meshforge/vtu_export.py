"""
Export Abaqus mesh (nodes + elements) to VTK UnstructuredGrid (.vtu) format.

Writes XML VTU so the mesh can be viewed in Paraview or other VTK-based tools.
Node IDs from the manager are 1-based; VTU uses 0-based point indices.
"""

import numpy as np
from pathlib import Path
from typing import Union

from meshforge.manager import AbaqusManager


# VTK cell type constants
VTK_TRIANGLE = 5
VTK_QUAD = 9
VTK_TETRA = 10
VTK_HEXAHEDRON = 12


def _node_id_to_index(manager: AbaqusManager) -> dict:
    """Build mapping from node ID (1-based) to 0-based index in manager.nodes order."""
    if manager.nodes is None:
        return {}
    return {int(nid): i for i, nid in enumerate(manager.nodes.ids)}


def _vtk_cell_type(n_nodes: int, _element_type: str = "") -> int:
    """Return VTK cell type from number of nodes per element."""
    if n_nodes == 3:
        return VTK_TRIANGLE
    if n_nodes == 4:
        return VTK_QUAD
    if n_nodes == 6:
        return VTK_TRIANGLE
    if n_nodes == 8:
        return VTK_QUAD  # 2D 8-node quad; for 3D hex use VTK_HEXAHEDRON
    if n_nodes == 10:
        return VTK_TETRA
    if n_nodes == 20:
        return VTK_HEXAHEDRON
    return VTK_QUAD if n_nodes >= 4 else VTK_TRIANGLE


def write_vtu(manager: AbaqusManager, vtu_path: Union[str, Path]) -> str:
    """
    Write the current mesh (nodes + elements) to a .vtu file.

    Args:
        manager: AbaqusManager with nodes and elements loaded.
        vtu_path: Path for the output .vtu file.

    Returns:
        Path to the written file as string.
    """
    path = Path(vtu_path)
    if manager.nodes is None:
        raise ValueError("Manager has no nodes")
    if manager.elements is None:
        raise ValueError("Manager has no elements")

    node_id_to_idx = _node_id_to_index(manager)
    points = manager.nodes.data  # (N, 2) or (N, 3)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])

    n_points = len(points)
    ids = manager.elements.ids
    data = manager.elements.data  # (n_elem, n_nodes_per_elem)

    # Build connectivity and offsets for VTK (0-based indices)
    connectivity = []
    offsets = []
    types = []
    offset = 0
    for i in range(len(ids)):
        conn = data[i]
        for node_id in conn:
            idx = node_id_to_idx.get(int(node_id))
            if idx is None:
                raise ValueError(f"Element references node {node_id} not in node list")
            connectivity.append(idx)
        offset += len(conn)
        offsets.append(offset)
        types.append(_vtk_cell_type(len(conn), getattr(manager.elements, "element_type", "")))

    conn_str = " ".join(str(c) for c in connectivity)
    off_str = " ".join(str(o) for o in offsets)
    types_str = " ".join(str(t) for t in types)
    points_str = " ".join(f"{x:.6g} {y:.6g} {z:.6g}" for x, y, z in points)

    xml = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{n_points}" NumberOfCells="{len(ids)}">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          {points_str}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          {conn_str}
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          {off_str}
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          {types_str}
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""
    path.write_text(xml, encoding="utf-8")
    return str(path)
