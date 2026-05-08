"""
Zero-Copy Data Pipeline.

Bridges MFEM C++ data structures to PyTorch tensors with minimal memory
copies. Three copy points are eliminated:

  1. MFEM mesh vertices → numpy
     Python-loop-per-vertex → bulk GetVerticesArray() or single ctypes frombuffer

  2. MFEM GridFunction DOFs → numpy displacement array
     Python-loop-per-DOF → GetDataArray() + single reshape

  3. numpy arrays → PyTorch tensors (training hot loop)
     torch.tensor() always copies → torch.from_numpy() shares CPU memory;
     only the unavoidable H2D DMA transfer remains

  4. FEMDataset.load(): np.load() → np.load(mmap_mode='r') (demand paging)

All functions degrade gracefully to safe fallbacks when PyMFEM internals
differ across versions.
"""

import ctypes
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MFEM mesh → numpy
# ─────────────────────────────────────────────────────────────────────────────

def mesh_vertices_to_numpy(mesh, space_dim: int) -> np.ndarray:
    """
    Extract mesh vertex coordinates as a numpy array without per-vertex loops.

    Fast path: mesh.GetVerticesArray() → 1-D float64 view of C++ buffer.
    Fallback:  per-vertex ctypes.frombuffer (eliminates inner d-loop).

    Args:
        mesh: PyMFEM Mesh object
        space_dim: Spatial dimension (2 or 3)

    Returns:
        (N_vertices, space_dim) float64 ndarray. May be a view into C++ memory
        on the fast path — do not modify in-place.
    """
    n_verts = mesh.GetNV()

    # Fast path: contiguous C++ buffer exposed as numpy array
    try:
        raw = mesh.GetVerticesArray()  # shape (N*space_dim,), float64
        if isinstance(raw, np.ndarray) and raw.size == n_verts * space_dim:
            return raw.reshape(n_verts, space_dim)
    except (AttributeError, Exception):
        pass

    # Fallback: per-vertex ctypes read, vectorised over components
    logger.debug("mesh_vertices_to_numpy: using ctypes fallback")
    out = np.empty((n_verts, space_dim), dtype=np.float64)
    for i in range(n_verts):
        vertex = mesh.GetVertex(i)
        ptr = ctypes.cast(int(vertex), ctypes.POINTER(ctypes.c_double))
        # frombuffer reads all components in one C call instead of a Python loop
        out[i] = np.frombuffer(
            (ctypes.c_double * space_dim).from_address(ctypes.addressof(ptr.contents)),
            dtype=np.float64,
            count=space_dim,
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. MFEM GridFunction → numpy displacement / scalar field
# ─────────────────────────────────────────────────────────────────────────────

def gridfunction_nodal_to_numpy(gf, mesh, dim: int) -> np.ndarray:
    """
    Extract GridFunction nodal values as a (N_nodes, dim) numpy array.

    Fast path: gf.GetDataArray() returns a 1-D view of the C++ DOF buffer.
    For standard MFEM H1 vector spaces the DOF layout is by-component:
      [u0_x, u1_x, ..., uN_x, u0_y, ..., uN_y]
    A single reshape + transpose produces (N, dim) without any Python loop.

    Fallback: build a DOF index array once per component and fancy-index.

    Args:
        gf: PyMFEM GridFunction (scalar or vector)
        mesh: PyMFEM Mesh (used for fallback DOF mapping)
        dim: Number of components (1 for scalar, 2 for 2-D displacement, etc.)

    Returns:
        (N_nodes, dim) float64 ndarray. Scalar fields are returned as (N, 1).
    """
    n_nodes = mesh.GetNV()

    # Fast path: contiguous C++ DOF buffer
    try:
        dof_array = gf.GetDataArray()  # 1-D float64 view, length = n_dofs
        if isinstance(dof_array, np.ndarray):
            if dim == 1:
                return dof_array[:n_nodes].reshape(n_nodes, 1)
            else:
                # Standard H1 by-component ordering: shape (dim, N) → (N, dim)
                expected = n_nodes * dim
                if dof_array.size >= expected:
                    return dof_array[:expected].reshape(dim, n_nodes).T.copy()
    except (AttributeError, Exception):
        pass

    # Fallback: build DOF index arrays, then fancy-index gf once per component
    logger.debug("gridfunction_nodal_to_numpy: using DOF-index fallback")
    fes = gf.FESpace()
    result = np.empty((n_nodes, dim), dtype=np.float64)

    for d in range(dim):
        dof_indices = np.array(
            [fes.DofToVDof(i, d) for i in range(n_nodes)], dtype=np.intp
        )
        # Access gf as a sequence once with vectorised indexing where possible
        try:
            data = gf.GetDataArray()
            result[:, d] = data[dof_indices]
        except (AttributeError, Exception):
            for i in range(n_nodes):
                result[i, d] = gf[dof_indices[i]]

    return result


def scalar_gridfunction_to_numpy(gf, n_nodes: int) -> np.ndarray:
    """
    Extract scalar GridFunction as a (N_nodes,) array.

    Uses GetDataArray() fast path if available; falls back to per-node indexing.
    """
    try:
        data = gf.GetDataArray()
        if isinstance(data, np.ndarray):
            return data[:n_nodes].copy()
    except (AttributeError, Exception):
        pass

    logger.debug("scalar_gridfunction_to_numpy: using per-node fallback")
    out = np.empty(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        out[i] = gf[i]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. numpy → PyTorch (hot-loop conversions)
# ─────────────────────────────────────────────────────────────────────────────

def as_float32(arr: np.ndarray) -> np.ndarray:
    """
    Return arr as a contiguous float32 array. Zero-cost if already float32.

    Args:
        arr: Input numpy array (any dtype)

    Returns:
        C-contiguous float32 array (may be the same object if no cast needed)
    """
    return np.ascontiguousarray(arr, dtype=np.float32)


def numpy_to_tensor(arr: np.ndarray, device, non_blocking: bool = True):
    """
    Convert a contiguous float32 numpy array to a PyTorch tensor.

    Uses torch.from_numpy() to avoid a CPU→CPU copy. Only the H2D DMA
    transfer (when device != 'cpu') is unavoidable.

    Args:
        arr: C-contiguous float32 numpy array
        device: torch.device target
        non_blocking: Allow asynchronous H2D transfer (default True)

    Returns:
        PyTorch tensor on device
    """
    import torch
    # Ensure contiguous float32 — if already correct, as_float32 is a no-op
    cpu_t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return cpu_t.to(device, non_blocking=non_blocking)


def preallocate_float32(arrays: list) -> list:
    """
    Cast a list of numpy arrays to float32 in-place (replaces list elements).

    Call once in prepare_data() to front-load the dtype conversion so the
    training hot loop pays zero casting cost.

    Args:
        arrays: List of numpy arrays (modified in-place)

    Returns:
        Same list with all elements cast to float32
    """
    for i, arr in enumerate(arrays):
        if arr.dtype != np.float32:
            arrays[i] = arr.astype(np.float32)
    return arrays
