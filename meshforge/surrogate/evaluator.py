"""
Surrogate model evaluator.

Analyzes surrogate model performance and identifies regions
where the model has high error or uncertainty, guiding
adaptive data generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import PredictionResult, SurrogateModel


@dataclass
class WeakRegion:
    """
    A region in parameter space where the surrogate is weak.

    Attributes:
        parameter_ranges: Dict mapping parameter name to (min, max) range
        metric: The metric indicating weakness (error, uncertainty, etc.)
        metric_value: Value of the metric in this region
        priority: Priority for data generation (higher = more important)
        sample_count: Number of existing samples in this region
        suggested_samples: Number of new samples recommended
    """
    parameter_ranges: Dict[str, Tuple[float, float]]
    metric: str
    metric_value: float
    priority: float = 1.0
    sample_count: int = 0
    suggested_samples: int = 5

    def contains(self, params: Dict[str, float]) -> bool:
        """Check if a parameter set falls within this region."""
        for name, (min_val, max_val) in self.parameter_ranges.items():
            if name in params:
                if not (min_val <= params[name] <= max_val):
                    return False
        return True

    def sample_uniform(self, n_samples: int = 1) -> List[Dict[str, float]]:
        """Generate uniform random samples within this region."""
        samples = []
        for _ in range(n_samples):
            sample = {}
            for name, (min_val, max_val) in self.parameter_ranges.items():
                sample[name] = np.random.uniform(min_val, max_val)
            samples.append(sample)
        return samples

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameter_ranges": self.parameter_ranges,
            "metric": self.metric,
            "metric_value": float(self.metric_value),
            "priority": float(self.priority),
            "sample_count": self.sample_count,
            "suggested_samples": self.suggested_samples,
        }


@dataclass
class UncertaintyAnalysis:
    """
    Analysis of model uncertainty across parameter space.

    Attributes:
        weak_regions: List of identified weak regions
        overall_uncertainty: Average uncertainty across test set
        max_uncertainty: Maximum uncertainty observed
        coverage_gaps: Regions with no training data
        metrics: Additional analysis metrics
    """
    weak_regions: List[WeakRegion] = field(default_factory=list)
    overall_uncertainty: float = 0.0
    max_uncertainty: float = 0.0
    coverage_gaps: List[WeakRegion] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def get_top_weak_regions(self, k: int = 5) -> List[WeakRegion]:
        """Get top k weak regions by priority."""
        sorted_regions = sorted(
            self.weak_regions,
            key=lambda r: r.priority,
            reverse=True
        )
        return sorted_regions[:k]

    def total_suggested_samples(self) -> int:
        """Get total number of suggested new samples."""
        return sum(r.suggested_samples for r in self.weak_regions)


class SurrogateEvaluator:
    """
    Evaluates surrogate model and identifies weak regions.

    Performs:
    1. Error analysis on validation data
    2. Uncertainty quantification across parameter space
    3. Coverage analysis for data gaps
    4. Priority ranking for adaptive sampling
    """

    def __init__(
        self,
        model: SurrogateModel,
        parameter_names: List[str],
        parameter_bounds: Dict[str, Tuple[float, float]]
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained surrogate model
            parameter_names: Names of input parameters
            parameter_bounds: Bounds for each parameter {name: (min, max)}
        """
        self.model = model
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds

    def evaluate_on_data(
        self,
        parameters: np.ndarray,
        coordinates: np.ndarray,
        true_outputs: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data.

        Args:
            parameters: Input parameters (N, n_params)
            coordinates: Query coordinates (N_points, coord_dim)
            true_outputs: True field values (N, N_points, output_dim)

        Returns:
            Dictionary of error metrics
        """
        predictions = self.model.predict(parameters, coordinates)
        return self.model.compute_error(
            predictions.values.flatten(),
            true_outputs.flatten()
        )

    def analyze_uncertainty(
        self,
        coordinates: np.ndarray,
        n_probe_samples: int = 100,
        uncertainty_threshold: float = 0.1,
        grid_resolution: int = 10
    ) -> UncertaintyAnalysis:
        """
        Analyze model uncertainty across parameter space.

        Args:
            coordinates: Fixed coordinates for prediction
            n_probe_samples: Number of random samples to probe
            uncertainty_threshold: Threshold for identifying weak regions
            grid_resolution: Resolution for grid-based analysis

        Returns:
            UncertaintyAnalysis with weak regions identified
        """
        weak_regions = []
        uncertainties = []

        # Generate probe samples in parameter space
        probe_params = self._generate_probe_samples(n_probe_samples)

        # Evaluate uncertainty at each probe point
        for params in probe_params:
            param_array = np.array([[params[name] for name in self.parameter_names]])
            result = self.model.predict(param_array, coordinates)

            if result.uncertainty is not None:
                uncertainty = np.mean(result.uncertainty)
                uncertainties.append(uncertainty)

                # Check if this is a weak region
                if uncertainty > uncertainty_threshold:
                    region = self._create_local_region(params, uncertainty)
                    weak_regions.append(region)

        # Grid-based coverage analysis
        coverage_gaps = self._find_coverage_gaps(grid_resolution)

        # Compute overall statistics
        if uncertainties:
            overall_uncertainty = np.mean(uncertainties)
            max_uncertainty = np.max(uncertainties)
        else:
            overall_uncertainty = 0.0
            max_uncertainty = 0.0

        # Merge nearby weak regions
        weak_regions = self._merge_nearby_regions(weak_regions)

        # Compute priorities
        for region in weak_regions:
            region.priority = self._compute_priority(region)

        return UncertaintyAnalysis(
            weak_regions=weak_regions,
            overall_uncertainty=overall_uncertainty,
            max_uncertainty=max_uncertainty,
            coverage_gaps=coverage_gaps,
            metrics={
                "n_probe_samples": n_probe_samples,
                "n_weak_regions": len(weak_regions),
                "n_coverage_gaps": len(coverage_gaps),
            }
        )

    def analyze_errors(
        self,
        parameters: np.ndarray,
        coordinates: np.ndarray,
        true_outputs: np.ndarray,
        error_threshold: float = 0.1
    ) -> UncertaintyAnalysis:
        """
        Analyze prediction errors to identify weak regions.

        Args:
            parameters: Input parameters (N, n_params)
            coordinates: Query coordinates
            true_outputs: True field values
            error_threshold: Threshold for identifying high-error regions

        Returns:
            UncertaintyAnalysis based on actual errors
        """
        weak_regions = []
        errors = []

        # Compute error for each sample
        for i in range(len(parameters)):
            param = parameters[i:i+1]
            true_out = true_outputs[i:i+1]

            pred = self.model.predict(param, coordinates)
            error = np.mean(np.abs(pred.values - true_out))
            errors.append(error)

            if error > error_threshold:
                params_dict = {
                    name: float(param[0, j])
                    for j, name in enumerate(self.parameter_names)
                }
                region = self._create_local_region(params_dict, error, metric="error")
                weak_regions.append(region)

        # Merge and prioritize
        weak_regions = self._merge_nearby_regions(weak_regions)
        for region in weak_regions:
            region.priority = self._compute_priority(region)

        return UncertaintyAnalysis(
            weak_regions=weak_regions,
            overall_uncertainty=np.mean(errors) if errors else 0.0,
            max_uncertainty=np.max(errors) if errors else 0.0,
            metrics={
                "mean_error": float(np.mean(errors)) if errors else 0.0,
                "max_error": float(np.max(errors)) if errors else 0.0,
                "n_high_error_samples": len(weak_regions),
            }
        )

    def _generate_probe_samples(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate random probe samples in parameter space."""
        samples = []
        for _ in range(n_samples):
            sample = {}
            for name in self.parameter_names:
                min_val, max_val = self.parameter_bounds[name]
                sample[name] = np.random.uniform(min_val, max_val)
            samples.append(sample)
        return samples

    def _create_local_region(
        self,
        center: Dict[str, float],
        metric_value: float,
        metric: str = "uncertainty",
        region_size: float = 0.1
    ) -> WeakRegion:
        """Create a weak region around a point."""
        ranges = {}
        for name in self.parameter_names:
            min_bound, max_bound = self.parameter_bounds[name]
            span = max_bound - min_bound
            half_size = span * region_size / 2

            center_val = center[name]
            ranges[name] = (
                max(min_bound, center_val - half_size),
                min(max_bound, center_val + half_size)
            )

        return WeakRegion(
            parameter_ranges=ranges,
            metric=metric,
            metric_value=metric_value,
        )

    def _find_coverage_gaps(self, resolution: int) -> List[WeakRegion]:
        """Find gaps in parameter space coverage using grid."""
        # This would require access to training data
        # For now, return empty list
        # TODO: Implement with training data access
        return []

    def _merge_nearby_regions(
        self,
        regions: List[WeakRegion],
        overlap_threshold: float = 0.5
    ) -> List[WeakRegion]:
        """Merge overlapping weak regions."""
        if not regions:
            return regions

        # Simple merging: keep distinct regions
        # TODO: Implement proper overlap detection and merging
        merged = []
        for region in regions:
            # Check if region overlaps significantly with existing
            is_redundant = False
            for existing in merged:
                overlap = self._compute_overlap(region, existing)
                if overlap > overlap_threshold:
                    # Keep the one with higher metric value
                    if region.metric_value > existing.metric_value:
                        merged.remove(existing)
                        merged.append(region)
                    is_redundant = True
                    break

            if not is_redundant:
                merged.append(region)

        return merged

    def _compute_overlap(self, region1: WeakRegion, region2: WeakRegion) -> float:
        """Compute overlap ratio between two regions."""
        overlap_volume = 1.0
        region1_volume = 1.0
        region2_volume = 1.0

        for name in self.parameter_names:
            r1_min, r1_max = region1.parameter_ranges.get(name, (0, 1))
            r2_min, r2_max = region2.parameter_ranges.get(name, (0, 1))

            # Overlap in this dimension
            overlap_min = max(r1_min, r2_min)
            overlap_max = min(r1_max, r2_max)
            overlap_size = max(0, overlap_max - overlap_min)

            overlap_volume *= overlap_size
            region1_volume *= (r1_max - r1_min)
            region2_volume *= (r2_max - r2_min)

        if region1_volume + region2_volume - overlap_volume < 1e-10:
            return 0.0

        # Jaccard-like overlap
        return overlap_volume / (region1_volume + region2_volume - overlap_volume)

    def _compute_priority(self, region: WeakRegion) -> float:
        """Compute sampling priority for a region."""
        # Higher priority for:
        # - Higher uncertainty/error
        # - Fewer existing samples
        # - Larger region size

        base_priority = region.metric_value

        # Penalize regions that already have samples
        sample_penalty = 1.0 / (1.0 + region.sample_count)

        # Bonus for larger regions (more potential impact)
        region_size = 1.0
        for name, (min_val, max_val) in region.parameter_ranges.items():
            bound_min, bound_max = self.parameter_bounds.get(name, (0, 1))
            relative_size = (max_val - min_val) / (bound_max - bound_min + 1e-10)
            region_size *= relative_size

        return base_priority * sample_penalty * (1 + region_size)

    def suggest_samples(
        self,
        analysis: UncertaintyAnalysis,
        budget: int = 10
    ) -> List[Dict[str, float]]:
        """
        Suggest new parameter samples based on analysis.

        Args:
            analysis: Uncertainty analysis results
            budget: Total number of samples to suggest

        Returns:
            List of parameter dictionaries for new simulations
        """
        suggestions = []

        # Get top weak regions
        top_regions = analysis.get_top_weak_regions(k=5)

        if not top_regions:
            # No weak regions, sample uniformly
            for _ in range(budget):
                sample = {}
                for name in self.parameter_names:
                    min_val, max_val = self.parameter_bounds[name]
                    sample[name] = np.random.uniform(min_val, max_val)
                suggestions.append(sample)
            return suggestions

        # Distribute budget across regions based on priority
        total_priority = sum(r.priority for r in top_regions)

        remaining_budget = budget
        for region in top_regions:
            if remaining_budget <= 0:
                break

            # Samples for this region proportional to priority
            n_samples = max(1, int(budget * region.priority / total_priority))
            n_samples = min(n_samples, remaining_budget)

            region_samples = region.sample_uniform(n_samples)
            suggestions.extend(region_samples)
            remaining_budget -= n_samples

        return suggestions[:budget]
