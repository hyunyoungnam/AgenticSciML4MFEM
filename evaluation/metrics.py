"""
Metrics calculation for FEA models.

Provides functions to calculate mesh quality metrics, convergence metrics,
and overall solution scores.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MeshQualityMetrics:
    """Mesh quality metrics for an FEA model."""
    num_nodes: int = 0
    num_elements: int = 0
    jacobian_min: Optional[float] = None
    jacobian_avg: Optional[float] = None
    jacobian_elements_below_threshold: int = 0
    aspect_ratio_min: Optional[float] = None
    aspect_ratio_max: Optional[float] = None
    aspect_ratio_avg: Optional[float] = None
    aspect_ratio_elements_above_threshold: int = 0
    warpage_max: Optional[float] = None
    skewness_max: Optional[float] = None
    min_edge_length: Optional[float] = None
    max_edge_length: Optional[float] = None
    edge_ratio_max: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "num_elements": self.num_elements,
            "jacobian_min": self.jacobian_min,
            "jacobian_avg": self.jacobian_avg,
            "jacobian_elements_below_threshold": self.jacobian_elements_below_threshold,
            "aspect_ratio_min": self.aspect_ratio_min,
            "aspect_ratio_max": self.aspect_ratio_max,
            "aspect_ratio_avg": self.aspect_ratio_avg,
            "aspect_ratio_elements_above_threshold": self.aspect_ratio_elements_above_threshold,
            "warpage_max": self.warpage_max,
            "skewness_max": self.skewness_max,
            "min_edge_length": self.min_edge_length,
            "max_edge_length": self.max_edge_length,
            "edge_ratio_max": self.edge_ratio_max,
        }

    def get_quality_score(
        self,
        jacobian_threshold: float = 0.1,
        aspect_ratio_threshold: float = 10.0,
    ) -> float:
        """
        Calculate overall mesh quality score.

        Args:
            jacobian_threshold: Minimum acceptable Jacobian
            aspect_ratio_threshold: Maximum acceptable aspect ratio

        Returns:
            Score between 0 and 1
        """
        score = 1.0

        # Jacobian contribution
        if self.jacobian_min is not None:
            if self.jacobian_min < jacobian_threshold:
                score -= 0.3 * (1 - self.jacobian_min / jacobian_threshold)
            elif self.jacobian_min < 0:
                score -= 0.5  # Negative Jacobian is very bad

        # Aspect ratio contribution
        if self.aspect_ratio_max is not None:
            if self.aspect_ratio_max > aspect_ratio_threshold:
                ratio_penalty = min(0.3, 0.1 * (self.aspect_ratio_max / aspect_ratio_threshold - 1))
                score -= ratio_penalty

        # Penalize elements with bad metrics
        if self.num_elements > 0:
            bad_jacobian_ratio = self.jacobian_elements_below_threshold / self.num_elements
            bad_ar_ratio = self.aspect_ratio_elements_above_threshold / self.num_elements
            score -= 0.1 * bad_jacobian_ratio
            score -= 0.1 * bad_ar_ratio

        return max(0.0, min(1.0, score))


class MetricsCalculator:
    """
    Calculator for FEA mesh quality metrics.

    Computes various mesh quality indicators from node coordinates
    and element connectivity.
    """

    def __init__(
        self,
        jacobian_threshold: float = 0.1,
        aspect_ratio_threshold: float = 10.0,
    ):
        """
        Initialize the metrics calculator.

        Args:
            jacobian_threshold: Minimum acceptable Jacobian
            aspect_ratio_threshold: Maximum acceptable aspect ratio
        """
        self.jacobian_threshold = jacobian_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold

    def calculate_from_manager(self, manager) -> MeshQualityMetrics:
        """
        Calculate mesh quality metrics from an AbaqusManager.

        Args:
            manager: AbaqusManager instance

        Returns:
            MeshQualityMetrics
        """
        metrics = MeshQualityMetrics()

        if manager.nodes is None or manager.elements is None:
            return metrics

        coords = manager.nodes.data
        node_ids = manager.nodes.ids
        elem_ids = manager.elements.ids
        connectivity = manager.elements.data

        metrics.num_nodes = len(node_ids)
        metrics.num_elements = len(elem_ids)

        # Calculate element-level metrics
        aspect_ratios = []
        jacobians = []
        edge_lengths = []

        for elem_idx in range(len(elem_ids)):
            elem_conn = connectivity[elem_idx]

            # Get element node coordinates
            elem_coords = self._get_element_coords(coords, node_ids, elem_conn)
            if elem_coords is None:
                continue

            # Calculate aspect ratio
            ar = self._calculate_aspect_ratio(elem_coords)
            if ar is not None:
                aspect_ratios.append(ar)

            # Calculate Jacobian (simplified for 2D quad elements)
            jac = self._calculate_jacobian_ratio(elem_coords)
            if jac is not None:
                jacobians.append(jac)

            # Calculate edge lengths
            edges = self._calculate_edge_lengths(elem_coords)
            edge_lengths.extend(edges)

        # Aggregate aspect ratio metrics
        if aspect_ratios:
            metrics.aspect_ratio_min = min(aspect_ratios)
            metrics.aspect_ratio_max = max(aspect_ratios)
            metrics.aspect_ratio_avg = sum(aspect_ratios) / len(aspect_ratios)
            metrics.aspect_ratio_elements_above_threshold = sum(
                1 for ar in aspect_ratios if ar > self.aspect_ratio_threshold
            )

        # Aggregate Jacobian metrics
        if jacobians:
            metrics.jacobian_min = min(jacobians)
            metrics.jacobian_avg = sum(jacobians) / len(jacobians)
            metrics.jacobian_elements_below_threshold = sum(
                1 for j in jacobians if j < self.jacobian_threshold
            )

        # Aggregate edge length metrics
        if edge_lengths:
            metrics.min_edge_length = min(edge_lengths)
            metrics.max_edge_length = max(edge_lengths)
            if metrics.min_edge_length > 0:
                metrics.edge_ratio_max = metrics.max_edge_length / metrics.min_edge_length

        return metrics

    def _get_element_coords(
        self,
        coords: np.ndarray,
        node_ids: np.ndarray,
        elem_conn: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Get coordinates for element nodes."""
        try:
            elem_coords = []
            for node_id in elem_conn:
                idx = np.where(node_ids == node_id)[0]
                if len(idx) > 0:
                    elem_coords.append(coords[idx[0]])
            return np.array(elem_coords) if len(elem_coords) >= 3 else None
        except Exception:
            return None

    def _calculate_aspect_ratio(self, elem_coords: np.ndarray) -> Optional[float]:
        """Calculate aspect ratio for an element."""
        try:
            # Use bounding box approach for simplicity
            ranges = np.ptp(elem_coords, axis=0)
            nonzero = ranges[ranges > 1e-10]
            if len(nonzero) >= 2:
                return float(max(nonzero) / min(nonzero))
            return None
        except Exception:
            return None

    def _calculate_jacobian_ratio(self, elem_coords: np.ndarray) -> Optional[float]:
        """
        Calculate Jacobian ratio for a 2D quadrilateral element.

        For a bilinear quad element, calculates the Jacobian at each corner
        and returns the minimum/maximum ratio.
        """
        try:
            if len(elem_coords) != 4:
                return None

            # Natural coordinates at corners
            corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

            jacobians = []
            for xi, eta in corners:
                # Shape function derivatives
                dN_dxi = np.array([
                    -(1 - eta) / 4,
                    (1 - eta) / 4,
                    (1 + eta) / 4,
                    -(1 + eta) / 4,
                ])
                dN_deta = np.array([
                    -(1 - xi) / 4,
                    -(1 + xi) / 4,
                    (1 + xi) / 4,
                    (1 - xi) / 4,
                ])

                # Jacobian matrix
                J = np.zeros((2, 2))
                for i in range(4):
                    J[0, 0] += dN_dxi[i] * elem_coords[i, 0]
                    J[0, 1] += dN_dxi[i] * elem_coords[i, 1]
                    J[1, 0] += dN_deta[i] * elem_coords[i, 0]
                    J[1, 1] += dN_deta[i] * elem_coords[i, 1]

                det_J = np.linalg.det(J)
                jacobians.append(det_J)

            if jacobians and max(abs(j) for j in jacobians) > 1e-10:
                # Return ratio of min/max (should be close to 1 for good elements)
                min_j = min(jacobians)
                max_j = max(jacobians)
                if max_j > 0:
                    return float(min_j / max_j)

            return None

        except Exception:
            return None

    def _calculate_edge_lengths(self, elem_coords: np.ndarray) -> List[float]:
        """Calculate edge lengths for an element."""
        edges = []
        try:
            n = len(elem_coords)
            for i in range(n):
                j = (i + 1) % n
                length = np.linalg.norm(elem_coords[i] - elem_coords[j])
                if length > 0:
                    edges.append(float(length))
        except Exception:
            pass
        return edges

    def calculate_convergence_metrics(
        self,
        solver_log: str,
    ) -> Dict[str, Any]:
        """
        Extract convergence metrics from solver log.

        Args:
            solver_log: Solver output log

        Returns:
            Dict with convergence metrics
        """
        import re

        metrics = {
            "converged": False,
            "iterations": None,
            "final_residual": None,
            "max_increment_cuts": 0,
            "warnings": [],
        }

        # Check for convergence
        if "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in solver_log:
            metrics["converged"] = True

        # Extract iteration count
        iter_match = re.search(r'ITERATION\s+(\d+)', solver_log, re.IGNORECASE)
        if iter_match:
            metrics["iterations"] = int(iter_match.group(1))

        # Extract residual
        residual_match = re.search(
            r'RESIDUAL\s*[:=]\s*([0-9.eE+-]+)',
            solver_log, re.IGNORECASE
        )
        if residual_match:
            try:
                metrics["final_residual"] = float(residual_match.group(1))
            except ValueError:
                pass

        # Count increment cutbacks
        cutback_count = len(re.findall(r'INCREMENT\s+CUT', solver_log, re.IGNORECASE))
        metrics["max_increment_cuts"] = cutback_count

        # Extract warnings
        warnings = re.findall(r'\*\*\*WARNING:(.+?)(?:\n|$)', solver_log)
        metrics["warnings"] = [w.strip() for w in warnings[:10]]

        return metrics


def calculate_solution_score(
    mesh_metrics: MeshQualityMetrics,
    preflight_score: float,
    convergence_metrics: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Calculate overall solution score.

    Args:
        mesh_metrics: Mesh quality metrics
        preflight_score: Pre-flight validation score
        convergence_metrics: Optional convergence metrics

    Returns:
        Score between 0 and 1
    """
    # Start with mesh quality
    mesh_score = mesh_metrics.get_quality_score()

    # Weight preflight score
    weighted_preflight = preflight_score * 0.3

    # Weight mesh score
    weighted_mesh = mesh_score * 0.3

    # Weight convergence
    if convergence_metrics is not None and convergence_metrics.get("converged"):
        conv_score = 0.4
        # Penalize for cutbacks
        cutbacks = convergence_metrics.get("max_increment_cuts", 0)
        conv_score -= 0.05 * min(cutbacks, 4)
    else:
        conv_score = 0.0 if convergence_metrics else 0.2  # Unknown convergence

    return min(1.0, weighted_preflight + weighted_mesh + conv_score)
