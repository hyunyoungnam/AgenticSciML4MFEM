"""
Acquisition functions for active learning.

Implements acquisition strategies to select the most informative
samples for surrogate model improvement.

Reference:
    Settles, B. (2009): "Active Learning Literature Survey"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np

from .base import PredictionResult, SurrogateModel


class AcquisitionType(Enum):
    """Types of acquisition functions."""
    UNCERTAINTY = auto()  # Pure uncertainty sampling
    EXPECTED_IMPROVEMENT = auto()  # Expected improvement over best
    QUERY_BY_COMMITTEE = auto()  # Ensemble disagreement


@dataclass
class AcquisitionResult:
    """Result of acquisition function evaluation."""
    scores: np.ndarray
    best_indices: np.ndarray
    metadata: Dict[str, Any]


class AcquisitionFunction(ABC):
    """
    Base class for acquisition functions.

    Uses template method pattern: subclasses implement _score_single()
    to score individual predictions, base class handles iteration.
    """

    def __init__(self, name: str = "base"):
        self.name = name

    def compute(
        self,
        candidates: np.ndarray,
        model: SurrogateModel,
        coordinates: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute acquisition values for candidate parameters.

        Template method - handles iteration, subclasses score.

        Args:
            candidates: Candidate parameter sets (N_candidates, n_params)
            model: Trained surrogate model with uncertainty estimates
            coordinates: Query coordinates for prediction
            **kwargs: Additional arguments for _score_single()

        Returns:
            Acquisition scores (N_candidates,) - higher = more informative
        """
        scores = np.zeros(len(candidates))
        for i, params in enumerate(candidates):
            result = model.predict(params.reshape(1, -1), coordinates)
            scores[i] = self._score_single(result, **kwargs)
        return scores

    @abstractmethod
    def _score_single(self, result: PredictionResult, **kwargs) -> float:
        """
        Compute score for single prediction. Override in subclass.

        Args:
            result: Prediction result from surrogate model
            **kwargs: Additional arguments

        Returns:
            Acquisition score (higher = more informative)
        """
        pass

    def select_batch(
        self,
        candidates: np.ndarray,
        model: SurrogateModel,
        coordinates: np.ndarray,
        batch_size: int,
        diversity_weight: float = 0.0,
        **kwargs
    ) -> AcquisitionResult:
        """
        Select a batch of candidates using the acquisition function.

        Args:
            candidates: All candidate parameter sets
            model: Trained surrogate model
            coordinates: Query coordinates
            batch_size: Number of candidates to select
            diversity_weight: Weight for diversity penalty (0 = no diversity)
            **kwargs: Additional arguments for compute()

        Returns:
            AcquisitionResult with selected indices and scores
        """
        scores = self.compute(candidates, model, coordinates, **kwargs)

        if diversity_weight > 0:
            selected_indices = self._diverse_selection(
                candidates, scores, batch_size, diversity_weight
            )
        else:
            selected_indices = np.argsort(scores)[-batch_size:][::-1]

        return AcquisitionResult(
            scores=scores,
            best_indices=selected_indices,
            metadata={
                "acquisition_type": self.name,
                "batch_size": batch_size,
                "diversity_weight": diversity_weight,
            }
        )

    def _diverse_selection(
        self,
        candidates: np.ndarray,
        scores: np.ndarray,
        batch_size: int,
        diversity_weight: float
    ) -> np.ndarray:
        """Greedy batch selection balancing acquisition score and diversity."""
        n_candidates = len(candidates)
        selected = []
        remaining_mask = np.ones(n_candidates, dtype=bool)

        # Normalize candidates for distance computation
        candidate_range = candidates.max(axis=0) - candidates.min(axis=0)
        candidate_range = np.where(candidate_range < 1e-10, 1.0, candidate_range)
        normalized = (candidates - candidates.min(axis=0)) / candidate_range

        for _ in range(min(batch_size, n_candidates)):
            if selected:
                selected_normalized = normalized[selected]
                distances = np.min(
                    np.linalg.norm(
                        normalized[:, None, :] - selected_normalized[None, :, :],
                        axis=2
                    ),
                    axis=1
                )
            else:
                distances = np.ones(n_candidates)

            combined = scores + diversity_weight * distances
            combined[~remaining_mask] = -np.inf

            best_idx = np.argmax(combined)
            selected.append(best_idx)
            remaining_mask[best_idx] = False

        return np.array(selected)


class UncertaintySampling(AcquisitionFunction):
    """
    Select points with highest predictive uncertainty.

    Simplest active learning strategy - sample where the model
    is most uncertain.
    """

    def __init__(self, aggregation: str = "mean"):
        """
        Args:
            aggregation: How to aggregate spatial uncertainty ("mean", "max", "median")
        """
        super().__init__(name="uncertainty")
        self.aggregation = aggregation

    def _score_single(self, result: PredictionResult, **kwargs) -> float:
        """Score based on prediction uncertainty."""
        if result.uncertainty is None:
            return 0.0

        uncertainty = result.uncertainty.flatten()
        if self.aggregation == "max":
            return float(np.max(uncertainty))
        elif self.aggregation == "median":
            return float(np.median(uncertainty))
        return float(np.mean(uncertainty))


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function.

    Balances exploration (uncertain regions) and exploitation
    (regions expected to have high error).
    """

    def __init__(self, xi: float = 0.01):
        """
        Args:
            xi: Exploration-exploitation tradeoff. Higher = more exploration.
        """
        super().__init__(name="expected_improvement")
        self.xi = xi

    def _score_single(
        self,
        result: PredictionResult,
        best_error: float = 0.0,
        **kwargs
    ) -> float:
        """Compute EI score."""
        from scipy.stats import norm

        mu = np.mean(np.abs(result.values))
        sigma = np.mean(result.uncertainty) if result.uncertainty is not None else 0.1

        if sigma > 1e-10:
            z = (best_error - mu - self.xi) / sigma
            ei = (best_error - mu - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)
            return float(max(0, ei))
        return float(max(0, best_error - mu))


class QueryByCommittee(AcquisitionFunction):
    """
    Query-by-Committee (QBC) acquisition function.

    Measures disagreement among ensemble members. Points where
    models disagree most are most informative.
    """

    def __init__(self, disagreement_metric: str = "variance"):
        """
        Args:
            disagreement_metric: How to measure disagreement ("variance", "range", "cv")
        """
        super().__init__(name="query_by_committee")
        self.disagreement_metric = disagreement_metric

    def compute(
        self,
        candidates: np.ndarray,
        model: SurrogateModel,
        coordinates: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Compute QBC scores based on ensemble disagreement."""
        # Check if model is an ensemble
        if not hasattr(model, '_models') or not model._models:
            # Fall back to uncertainty sampling
            return UncertaintySampling().compute(candidates, model, coordinates)

        scores = []
        for i in range(len(candidates)):
            params = candidates[i:i+1]

            # Get predictions from each ensemble member
            predictions = []
            for member in model._models:
                result = member.predict(params, coordinates)
                predictions.append(result.values.flatten())

            predictions = np.array(predictions)  # (n_models, n_points)

            # Compute disagreement
            if self.disagreement_metric == "range":
                disagreement = np.mean(np.ptp(predictions, axis=0))
            elif self.disagreement_metric == "cv":
                mean_pred = np.mean(predictions, axis=0)
                std_pred = np.std(predictions, axis=0)
                cv = std_pred / (np.abs(mean_pred) + 1e-10)
                disagreement = np.mean(cv)
            else:  # variance (default)
                disagreement = np.mean(np.var(predictions, axis=0))

            scores.append(disagreement)

        return np.array(scores)

    def _score_single(self, result: PredictionResult, **kwargs) -> float:
        """Not used for QBC - compute() handles ensemble logic."""
        return 0.0


def get_acquisition_function(
    acquisition_type: str,
    **kwargs
) -> AcquisitionFunction:
    """
    Factory function to create acquisition functions.

    Args:
        acquisition_type: Type name ("uncertainty", "ei", "qbc")
        **kwargs: Arguments for the specific acquisition function

    Returns:
        AcquisitionFunction instance
    """
    strategies = {
        "uncertainty": UncertaintySampling,
        "ei": ExpectedImprovement,
        "expected_improvement": ExpectedImprovement,
        "qbc": QueryByCommittee,
        "query_by_committee": QueryByCommittee,
    }

    # Handle enum input
    if hasattr(acquisition_type, 'name'):
        acquisition_type = acquisition_type.name.lower()
    else:
        acquisition_type = acquisition_type.lower()

    if acquisition_type not in strategies:
        raise ValueError(f"Unknown: {acquisition_type}. Choose from {list(strategies.keys())}")

    return strategies[acquisition_type](**kwargs)
