"""
Spatial error analysis for surrogate models.

Provides tools to analyze WHERE in the domain the surrogate model
has high errors, enabling targeted mesh refinement and intelligent
sampling strategies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import PredictionResult, SurrogateModel


@dataclass
class ErrorHotspot:
    """
    A region in spatial domain with concentrated errors.

    Attributes:
        center: Centroid of the hotspot (spatial coordinates)
        radius: Approximate radius of the region
        mean_error: Mean error in this region
        max_error: Maximum error in this region
        point_indices: Indices of mesh points in this hotspot
        parameter_sensitivity: How sensitive this region is to parameters
    """
    center: np.ndarray
    radius: float
    mean_error: float
    max_error: float
    point_indices: np.ndarray
    parameter_sensitivity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialErrorAnalysis:
    """
    Result of spatial error analysis.

    Attributes:
        error_field: Per-point error values
        hotspots: Identified error hotspots
        global_stats: Global error statistics
        parameter_influence: Influence of each parameter on spatial error
    """
    error_field: np.ndarray
    hotspots: List[ErrorHotspot] = field(default_factory=list)
    global_stats: Dict[str, float] = field(default_factory=dict)
    parameter_influence: Dict[str, np.ndarray] = field(default_factory=dict)


class SpatialErrorAnalyzer:
    """
    Analyzes spatial distribution of surrogate model errors.

    Identifies regions in the physical domain where the surrogate
    has high prediction error, which can inform:
    1. Mesh refinement strategies
    2. Feature engineering for better spatial representation
    3. Understanding of model limitations
    """

    def __init__(
        self,
        model: SurrogateModel,
        coordinates: np.ndarray
    ):
        """
        Initialize analyzer.

        Args:
            model: Trained surrogate model
            coordinates: Mesh coordinates (N_points, coord_dim)
        """
        self.model = model
        self.coordinates = coordinates
        self.coord_dim = coordinates.shape[1]

    def compute_error_field(
        self,
        params: np.ndarray,
        true_values: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-point error field for given parameters.

        Args:
            params: Parameter values (1, n_params) or (n_params,)
            true_values: Ground truth field values (N_points,) or (N_points, dim)

        Returns:
            Error field (N_points,)
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)

        # Get prediction
        result = self.model.predict(params, self.coordinates)
        predictions = result.values.flatten()

        # Compute absolute error
        true_flat = true_values.flatten()
        if len(predictions) != len(true_flat):
            # Handle dimension mismatch
            min_len = min(len(predictions), len(true_flat))
            predictions = predictions[:min_len]
            true_flat = true_flat[:min_len]

        error_field = np.abs(predictions - true_flat)

        return error_field

    def compute_relative_error_field(
        self,
        params: np.ndarray,
        true_values: np.ndarray,
        epsilon: float = 1e-10
    ) -> np.ndarray:
        """
        Compute per-point relative error field.

        Args:
            params: Parameter values
            true_values: Ground truth field values
            epsilon: Small value to avoid division by zero

        Returns:
            Relative error field (N_points,)
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)

        result = self.model.predict(params, self.coordinates)
        predictions = result.values.flatten()
        true_flat = true_values.flatten()

        # Handle dimension mismatch
        if len(predictions) != len(true_flat):
            min_len = min(len(predictions), len(true_flat))
            predictions = predictions[:min_len]
            true_flat = true_flat[:min_len]

        relative_error = np.abs(predictions - true_flat) / (np.abs(true_flat) + epsilon)

        return relative_error

    def identify_hotspots(
        self,
        error_field: np.ndarray,
        threshold_percentile: float = 90,
        min_cluster_size: int = 3,
        clustering_eps: Optional[float] = None
    ) -> List[ErrorHotspot]:
        """
        Find spatial regions with concentrated high errors.

        Args:
            error_field: Per-point error values
            threshold_percentile: Percentile threshold for high error
            min_cluster_size: Minimum points to form a hotspot
            clustering_eps: DBSCAN epsilon (auto-computed if None)

        Returns:
            List of ErrorHotspot objects
        """
        # Identify high-error points
        threshold = np.percentile(error_field, threshold_percentile)
        high_error_mask = error_field > threshold
        high_error_indices = np.where(high_error_mask)[0]

        if len(high_error_indices) < min_cluster_size:
            return []

        high_error_coords = self.coordinates[high_error_indices]
        high_error_values = error_field[high_error_indices]

        # Cluster high-error points
        if clustering_eps is None:
            # Auto-compute based on coordinate spread
            coord_range = self.coordinates.max(axis=0) - self.coordinates.min(axis=0)
            clustering_eps = np.mean(coord_range) * 0.1

        hotspots = self._cluster_points(
            high_error_coords,
            high_error_values,
            high_error_indices,
            clustering_eps,
            min_cluster_size
        )

        return hotspots

    def _cluster_points(
        self,
        coords: np.ndarray,
        errors: np.ndarray,
        indices: np.ndarray,
        eps: float,
        min_samples: int
    ) -> List[ErrorHotspot]:
        """Cluster high-error points into hotspots."""
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            labels = clustering.labels_
        except ImportError:
            # Fallback to simple grid-based clustering
            labels = self._simple_clustering(coords, eps)

        hotspots = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        for label in unique_labels:
            mask = labels == label
            cluster_coords = coords[mask]
            cluster_errors = errors[mask]
            cluster_indices = indices[mask]

            center = np.mean(cluster_coords, axis=0)
            radius = np.max(np.linalg.norm(cluster_coords - center, axis=1))

            hotspot = ErrorHotspot(
                center=center,
                radius=radius,
                mean_error=float(np.mean(cluster_errors)),
                max_error=float(np.max(cluster_errors)),
                point_indices=cluster_indices,
            )
            hotspots.append(hotspot)

        # Sort by mean error (highest first)
        hotspots.sort(key=lambda h: h.mean_error, reverse=True)

        return hotspots

    def _simple_clustering(
        self,
        coords: np.ndarray,
        eps: float
    ) -> np.ndarray:
        """Simple fallback clustering without sklearn."""
        n_points = len(coords)
        labels = -np.ones(n_points, dtype=int)
        current_label = 0

        for i in range(n_points):
            if labels[i] != -1:
                continue

            # Find neighbors
            distances = np.linalg.norm(coords - coords[i], axis=1)
            neighbors = np.where(distances < eps)[0]

            if len(neighbors) >= 2:
                labels[neighbors] = current_label
                current_label += 1

        return labels

    def compute_parameter_sensitivity(
        self,
        base_params: np.ndarray,
        parameter_names: List[str],
        epsilon: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """
        Compute sensitivity of spatial error to each parameter.

        Uses finite differences to estimate how much the error field
        changes when each parameter is perturbed.

        Args:
            base_params: Base parameter values (n_params,)
            parameter_names: Names of parameters
            epsilon: Perturbation size (relative)

        Returns:
            Dict mapping parameter name to sensitivity field (N_points,)
        """
        if base_params.ndim == 1:
            base_params = base_params.reshape(1, -1)

        base_result = self.model.predict(base_params, self.coordinates)
        base_values = base_result.values.flatten()

        sensitivities = {}

        for i, name in enumerate(parameter_names):
            # Perturb parameter
            params_plus = base_params.copy()
            param_range = abs(base_params[0, i]) if base_params[0, i] != 0 else 1.0
            delta = epsilon * param_range

            params_plus[0, i] += delta

            # Get perturbed prediction
            result_plus = self.model.predict(params_plus, self.coordinates)
            values_plus = result_plus.values.flatten()

            # Compute sensitivity (derivative magnitude)
            if len(values_plus) != len(base_values):
                min_len = min(len(values_plus), len(base_values))
                sensitivity = np.abs(values_plus[:min_len] - base_values[:min_len]) / delta
            else:
                sensitivity = np.abs(values_plus - base_values) / delta

            sensitivities[name] = sensitivity

        return sensitivities

    def analyze(
        self,
        params: np.ndarray,
        true_values: np.ndarray,
        parameter_names: Optional[List[str]] = None,
        hotspot_threshold: float = 90,
    ) -> SpatialErrorAnalysis:
        """
        Perform comprehensive spatial error analysis.

        Args:
            params: Parameter values
            true_values: Ground truth field values
            parameter_names: Names of parameters for sensitivity analysis
            hotspot_threshold: Percentile threshold for hotspot detection

        Returns:
            SpatialErrorAnalysis with all results
        """
        # Compute error field
        error_field = self.compute_error_field(params, true_values)

        # Identify hotspots
        hotspots = self.identify_hotspots(error_field, threshold_percentile=hotspot_threshold)

        # Global statistics
        global_stats = {
            "mean_error": float(np.mean(error_field)),
            "max_error": float(np.max(error_field)),
            "std_error": float(np.std(error_field)),
            "median_error": float(np.median(error_field)),
            "n_hotspots": len(hotspots),
            "hotspot_coverage": sum(len(h.point_indices) for h in hotspots) / len(error_field),
        }

        # Parameter sensitivity
        param_influence = {}
        if parameter_names:
            param_influence = self.compute_parameter_sensitivity(
                params, parameter_names
            )

            # Add sensitivity info to hotspots
            for hotspot in hotspots:
                total_sensitivity = 0.0
                for name, sensitivity_field in param_influence.items():
                    if len(sensitivity_field) > max(hotspot.point_indices):
                        hotspot_sensitivity = np.mean(sensitivity_field[hotspot.point_indices])
                        total_sensitivity += hotspot_sensitivity
                hotspot.parameter_sensitivity = total_sensitivity

        return SpatialErrorAnalysis(
            error_field=error_field,
            hotspots=hotspots,
            global_stats=global_stats,
            parameter_influence=param_influence,
        )

    def aggregate_analyses(
        self,
        analyses: List[SpatialErrorAnalysis]
    ) -> SpatialErrorAnalysis:
        """
        Aggregate multiple spatial analyses into one.

        Useful for combining analyses across multiple parameter samples.

        Args:
            analyses: List of SpatialErrorAnalysis from different samples

        Returns:
            Aggregated analysis
        """
        if not analyses:
            return SpatialErrorAnalysis(error_field=np.array([]))

        # Stack error fields
        error_fields = np.stack([a.error_field for a in analyses])
        mean_error_field = np.mean(error_fields, axis=0)
        max_error_field = np.max(error_fields, axis=0)

        # Combine hotspots (use max error field for detection)
        hotspots = self.identify_hotspots(max_error_field)

        # Aggregate statistics
        global_stats = {
            "mean_error": float(np.mean(mean_error_field)),
            "max_error": float(np.max(max_error_field)),
            "std_error": float(np.std(mean_error_field)),
            "n_samples": len(analyses),
            "n_hotspots": len(hotspots),
        }

        return SpatialErrorAnalysis(
            error_field=mean_error_field,
            hotspots=hotspots,
            global_stats=global_stats,
        )


class ErrorDecomposer:
    """
    Decomposes surrogate model error into interpretable components.

    Provides insight into error sources:
    - Bias (systematic under/over prediction)
    - Variance (inconsistency across ensemble)
    - Noise (random error component)
    """

    def __init__(self, model: SurrogateModel):
        """
        Initialize decomposer.

        Args:
            model: Surrogate model (preferably ensemble for variance)
        """
        self.model = model
        self._is_ensemble = hasattr(model, '_models')

    def decompose(
        self,
        params: np.ndarray,
        coordinates: np.ndarray,
        true_values: np.ndarray
    ) -> Dict[str, float]:
        """
        Decompose total error into bias, variance, and noise.

        Uses bias-variance decomposition:
        MSE = Bias^2 + Variance + Noise

        Args:
            params: Parameter values
            coordinates: Query coordinates
            true_values: Ground truth values

        Returns:
            Dictionary with bias, variance, noise, and total MSE
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)

        result = self.model.predict(params, coordinates)
        predictions = result.values.flatten()
        true_flat = true_values.flatten()

        # Handle dimension mismatch
        if len(predictions) != len(true_flat):
            min_len = min(len(predictions), len(true_flat))
            predictions = predictions[:min_len]
            true_flat = true_flat[:min_len]

        # Bias: systematic error (mean prediction - mean true)
        bias = np.mean(predictions) - np.mean(true_flat)
        bias_squared = bias ** 2

        # For ensemble models, compute variance across members
        if self._is_ensemble and result.uncertainty is not None:
            uncertainty = result.uncertainty.flatten()
            if len(uncertainty) != len(predictions):
                uncertainty = uncertainty[:len(predictions)]
            variance = np.mean(uncertainty ** 2)
        else:
            variance = 0.0

        # Total MSE
        mse = np.mean((predictions - true_flat) ** 2)

        # Noise estimate (residual)
        noise = max(0, mse - bias_squared - variance)

        return {
            "bias": float(bias),
            "bias_squared": float(bias_squared),
            "variance": float(variance),
            "noise": float(noise),
            "mse": float(mse),
            "bias_fraction": float(bias_squared / (mse + 1e-10)),
            "variance_fraction": float(variance / (mse + 1e-10)),
        }

    def decompose_spatial(
        self,
        params: np.ndarray,
        coordinates: np.ndarray,
        true_values: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute spatial decomposition of error.

        Returns per-point bias and variance.

        Args:
            params: Parameter values
            coordinates: Query coordinates
            true_values: Ground truth values

        Returns:
            Dictionary with spatial bias and variance fields
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)

        result = self.model.predict(params, coordinates)
        predictions = result.values.flatten()
        true_flat = true_values.flatten()

        # Handle dimension mismatch
        min_len = min(len(predictions), len(true_flat))
        predictions = predictions[:min_len]
        true_flat = true_flat[:min_len]

        # Spatial bias (signed error)
        spatial_bias = predictions - true_flat

        # Spatial variance (from ensemble)
        if self._is_ensemble and result.uncertainty is not None:
            spatial_variance = result.uncertainty.flatten()[:min_len]
        else:
            spatial_variance = np.zeros_like(spatial_bias)

        # Spatial MSE
        spatial_mse = spatial_bias ** 2

        return {
            "bias": spatial_bias,
            "variance": spatial_variance,
            "mse": spatial_mse,
            "signed_error": spatial_bias,
            "absolute_error": np.abs(spatial_bias),
        }
