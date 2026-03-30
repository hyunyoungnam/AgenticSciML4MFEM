"""
H-refinement (element splitting) for error-driven mesh adaptation.

Implements h-refinement by splitting elements in high-error regions.
Unlike r-adaptivity (which moves nodes), h-refinement adds new nodes
and elements where the error is largest.

Key concepts:
1. Per-element error field drives which elements get split.
2. Elements with error above threshold_fraction * max_error are refined.
3. Multiple levels of refinement can be applied recursively.
4. After each level, the error field is re-interpolated to the new mesh
   using nearest-centroid mapping.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..mesh.mfem_manager import MFEMManager


@dataclass
class HRefinementConfig:
    """
    Configuration for h-refinement.

    Attributes:
        error_threshold: Fraction of max error above which elements are split
                         (0.5 = elements with error > 0.5 * max_error get split)
        max_refinement_levels: Maximum recursive refinement depth
        min_elements: Do not refine if the mesh already has this many elements
        max_elements: Stop refining if element count would exceed this
        nonconforming: Passed to MFEM (-1=auto, 0=conforming, 1=nonconforming)
    """
    error_threshold: float = 0.5
    max_refinement_levels: int = 2
    min_elements: int = 50
    max_elements: int = 2000
    nonconforming: int = -1


@dataclass
class HRefinementResult:
    """
    Result of h-refinement.

    Attributes:
        success: Whether refinement succeeded (at least one level applied)
        num_elements_before: Element count before any refinement
        num_elements_after: Element count after all refinement levels
        num_nodes_before: Node count before any refinement
        num_nodes_after: Node count after all refinement levels
        levels_applied: Number of refinement levels actually performed
        error_message: Error message if something went wrong
    """
    success: bool
    num_elements_before: int
    num_elements_after: int
    num_nodes_before: int
    num_nodes_after: int
    levels_applied: int
    error_message: Optional[str] = None


class HRefinement:
    """
    H-refinement engine that splits elements in high-error regions.

    Uses MFEM's RefineByError to split elements whose per-element error
    exceeds a fraction of the maximum error.  Multiple refinement levels
    can be applied, with the error field re-interpolated between levels.

    Example:
        error_field = surrogate.compute_element_error(manager)

        config = HRefinementConfig(error_threshold=0.4, max_refinement_levels=2)
        refiner = HRefinement(config)
        result = refiner.refine(manager, error_field)

        if result.success:
            print(f"Elements: {result.num_elements_before} -> {result.num_elements_after}")
    """

    def __init__(self, config: Optional[HRefinementConfig] = None):
        """
        Initialize the h-refinement engine.

        Args:
            config: Refinement configuration.  Uses defaults if None.
        """
        self.config = config or HRefinementConfig()

    def refine(self, manager: MFEMManager, error_field: np.ndarray) -> HRefinementResult:
        """
        Refine mesh by splitting elements with high error.

        Args:
            manager: MFEMManager with loaded mesh
            error_field: (N_elements,) array of per-element error values

        Returns:
            HRefinementResult
        """
        try:
            import mfem.ser as mfem
        except ImportError:
            return HRefinementResult(
                success=False,
                num_elements_before=0,
                num_elements_after=0,
                num_nodes_before=0,
                num_nodes_after=0,
                levels_applied=0,
                error_message="PyMFEM not installed. Install with: pip install mfem",
            )

        ne_before = manager.num_elements
        nv_before = manager.num_nodes

        # Validate error_field length — must be element-centered
        if len(error_field) != ne_before:
            return HRefinementResult(
                success=False,
                num_elements_before=ne_before,
                num_elements_after=ne_before,
                num_nodes_before=nv_before,
                num_nodes_after=nv_before,
                levels_applied=0,
                error_message=(
                    f"error_field length {len(error_field)} != "
                    f"num_elements {ne_before}"
                ),
            )

        # Guard: skip if already too many elements
        if ne_before >= self.config.max_elements:
            return HRefinementResult(
                success=False,
                num_elements_before=ne_before,
                num_elements_after=ne_before,
                num_nodes_before=nv_before,
                num_nodes_after=nv_before,
                levels_applied=0,
                error_message=(
                    f"Already at max_elements ({ne_before} >= "
                    f"{self.config.max_elements}); no refinement performed"
                ),
            )

        current_error = np.array(error_field, dtype=np.float64)
        levels_applied = 0

        for level in range(self.config.max_refinement_levels):
            # Compute element centers before refinement (used for interpolation)
            old_centers = self._get_element_centers(manager)

            max_err = np.max(current_error)
            if max_err <= 0.0:
                # No positive error — nothing to refine
                break

            threshold = float(self.config.error_threshold * max_err)

            # Build mfem.Vector from current error field
            err_vec = mfem.Vector(len(current_error))
            for i, v in enumerate(current_error):
                err_vec[i] = float(v)

            refined = manager._mesh.RefineByError(
                err_vec, threshold, self.config.nonconforming
            )

            if not refined:
                # MFEM found nothing to split at this threshold
                break

            # Sync Python-side caches after MFEM modified the mesh in place
            manager._extract_mesh_data()
            levels_applied += 1

            # Stop if we've hit the element cap
            if manager.num_elements >= self.config.max_elements:
                break

            # Re-interpolate error to the new element count for the next level
            if level < self.config.max_refinement_levels - 1:
                new_centers = self._get_element_centers(manager)
                current_error = _interpolate_error_to_refined(
                    old_centers, current_error, new_centers
                )

        return HRefinementResult(
            success=levels_applied > 0,
            num_elements_before=ne_before,
            num_elements_after=manager.num_elements,
            num_nodes_before=nv_before,
            num_nodes_after=manager.num_nodes,
            levels_applied=levels_applied,
        )

    def _get_element_centers(self, manager: MFEMManager) -> np.ndarray:
        """
        Compute centroid coordinates for every element.

        Returns:
            (N_elements, dim) array of element centers
        """
        elements = manager.get_elements()   # (N, max_nodes), -1 = padding
        nodes = manager.get_nodes()         # (V, dim)
        centers = np.zeros((len(elements), nodes.shape[1]), dtype=np.float64)
        for i, elem in enumerate(elements):
            valid = elem[elem >= 0]
            centers[i] = nodes[valid].mean(axis=0)
        return centers


def _interpolate_error_to_refined(
    old_centers: np.ndarray,
    old_error: np.ndarray,
    new_centers: np.ndarray,
) -> np.ndarray:
    """
    Map error values from the old element set to the new (refined) element set.

    For each new element center, the nearest old element center is found and
    its error value is assigned.  Uses pure numpy (no scipy required).

    Args:
        old_centers: (N_old, dim) array of old element centroids
        old_error:   (N_old,) array of per-element error values on old mesh
        new_centers: (N_new, dim) array of new element centroids

    Returns:
        (N_new,) array of interpolated error values
    """
    # Squared distances: (N_new, N_old)
    diff = new_centers[:, np.newaxis, :] - old_centers[np.newaxis, :, :]  # (N_new, N_old, dim)
    sq_dist = np.sum(diff ** 2, axis=-1)  # (N_new, N_old)
    nearest = np.argmin(sq_dist, axis=1)  # (N_new,)
    return old_error[nearest]


def is_h_refinement_available() -> bool:
    """
    Check whether PyMFEM with h-refinement support is available.

    Returns:
        True if RefineByError and GeneralRefinement exist on mfem.Mesh
    """
    try:
        import mfem.ser as mfem
        return (
            hasattr(mfem.Mesh, "RefineByError")
            and hasattr(mfem.Mesh, "GeneralRefinement")
        )
    except ImportError:
        return False
