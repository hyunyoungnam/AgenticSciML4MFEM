"""
Acquisition functions for active learning.

Implements various acquisition strategies to select the most informative
samples for surrogate model improvement. These functions quantify the
"informativeness" of a candidate sample based on model uncertainty,
expected error reduction, or ensemble disagreement.

Reference:
    Settles, B. (2009): "Active Learning Literature Survey"
    Snoek et al. (2012): "Practical Bayesian Optimization of ML Algorithms"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import PredictionResult, SurrogateModel


class AcquisitionType(Enum):
    """Types of acquisition functions."""
    UNCERTAINTY = auto()  # Pure uncertainty sampling
    EXPECTED_IMPROVEMENT = auto()  # Expected improvement over best
    QUERY_BY_COMMITTEE = auto()  # Ensemble disagreement
    UCB = auto()  # Upper confidence bound
    PROBABILITY_OF_IMPROVEMENT = auto()  # Probability of improving best
    GREEDY = auto()  # Highest predicted error
    HYBRID = auto()  # Weighted combination


@dataclass
class AcquisitionResult:
    """
    Result of acquisition function evaluation.

    Attributes:
        scores: Acquisition scores for each candidate
        best_indices: Indices of top candidates
        metadata: Additional information
    """
    scores: np.ndarray
    best_indices: np.ndarray
    metadata: Dict[str, Any]


class AcquisitionFunction(ABC):
    """
    Base class for acquisition functions.

    An acquisition function evaluates candidate parameter configurations
    and returns a score indicating how "informative" each candidate would be
    if we were to run a simulation there.
    """

    def __init__(self, name: str = "base"):
        self.name = name

    @abstractmethod
    def compute(
        self,
        candidates: np.ndarray,
        model: SurrogateModel,
        coordinates: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute acquisition values for candidate parameters.

        Args:
            candidates: Candidate parameter sets (N_candidates, n_params)
            model: Trained surrogate model with uncertainty estimates
            coordinates: Query coordinates for prediction
            **kwargs: Additional arguments specific to acquisition function

        Returns:
            Acquisition scores (N_candidates,) - higher = more informative
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
        # Compute acquisition scores
        scores = self.compute(candidates, model, coordinates, **kwargs)

        if diversity_weight > 0:
            # Greedy selection with diversity
            selected_indices = self._diverse_selection(
                candidates, scores, batch_size, diversity_weight
            )
        else:
            # Pure greedy selection by score
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
        """
        Greedy batch selection balancing acquisition score and diversity.

        Uses a greedy algorithm that penalizes candidates close to
        already-selected candidates.
        """
        n_candidates = len(candidates)
        selected = []
        remaining_mask = np.ones(n_candidates, dtype=bool)

        # Normalize candidates for distance computation
        candidate_range = candidates.max(axis=0) - candidates.min(axis=0)
        candidate_range = np.where(candidate_range < 1e-10, 1.0, candidate_range)
        normalized = (candidates - candidates.min(axis=0)) / candidate_range

        for _ in range(min(batch_size, n_candidates)):
            # Compute diversity penalty
            if selected:
                selected_normalized = normalized[selected]
                # Min distance to any selected point
                distances = np.min(
                    np.linalg.norm(
                        normalized[:, None, :] - selected_normalized[None, :, :],
                        axis=2
                    ),
                    axis=1
                )
            else:
                distances = np.ones(n_candidates)

            # Combined score: acquisition + diversity
            combined = scores + diversity_weight * distances
            combined[~remaining_mask] = -np.inf

            # Select best
            best_idx = np.argmax(combined)
            selected.append(best_idx)
            remaining_mask[best_idx] = False

        return np.array(selected)


class UncertaintySampling(AcquisitionFunction):
    """
    Select points with highest predictive uncertainty.

    This is the simplest active learning strategy - sample where the model
    is most uncertain. Requires a model that provides uncertainty estimates
    (e.g., ensemble).

    Uncertainty can be measured as:
    - Standard deviation across ensemble predictions
    - Variance of predictions
    - Entropy (for classification)
    """

    def __init__(self, aggregation: str = "mean"):
        """
        Initialize uncertainty sampling.

        Args:
            aggregation: How to aggregate spatial uncertainty
                        ("mean", "max", "median")
        """
        super().__init__(name="uncertainty")
        self.aggregation = aggregation

    def compute(
        self,
        candidates: np.ndarray,
        model: SurrogateModel,
        coordinates: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Compute uncertainty scores for candidates."""
        scores = []

        for i in range(len(candidates)):
            params = candidates[i:i+1]
            result = model.predict(params, coordinates)

            if result.uncertainty is not None:
                uncertainty = result.uncertainty.flatten()
                if self.aggregation == "mean":
                    score = np.mean(uncertainty)
                elif self.aggregation == "max":
                    score = np.max(uncertainty)
                elif self.aggregation == "median":
                    score = np.median(uncertainty)
                else:
                    score = np.mean(uncertainty)
            else:
                # No uncertainty available - use default
                score = 0.0

            scores.append(score)

        return np.array(scores)


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function.

    Balances exploration (sampling uncertain regions) and exploitation
    (sampling regions expected to have high error). Commonly used in
    Bayesian optimization.

    EI = E[max(0, f(x) - f_best)] where f(x) is the model error

    For error minimization:
    EI(x) = (f_best - mu(x) - xi) * Phi(Z) + sigma(x) * phi(Z)
    where Z = (f_best - mu(x) - xi) / sigma(x)
    """

    def __init__(self, xi: float = 0.01):
        """
        Initialize EI acquisition.

        Args:
            xi: Exploration-exploitation tradeoff parameter.
                Higher values favor exploration.
        """
        super().__init__(name="expected_improvement")
        self.xi = xi

    def compute(
        self,
        candidates: np.ndarray,
        model: SurrogateModel,
        coordinates: np.ndarray,
        best_error: float = 0.0,
        **kwargs
    ) -> np.ndarray:
        """
        Compute EI scores for candidates.

        Args:
            candidates: Candidate parameters
            model: Surrogate model
            coordinates: Query coordinates
            best_error: Best (lowest) error observed so far
        """
        from scipy.stats import norm

        scores = []

        for i in range(len(candidates)):
            params = candidates[i:i+1]
            result = model.predict(params, coordinates)

            # Expected error (mean prediction magnitude as proxy for error)
            mu = np.mean(np.abs(result.values))

            # Uncertainty
            if result.uncertainty is not None:
                sigma = np.mean(result.uncertainty)
            else:
                sigma = 0.1  # Default uncertainty

            # Compute EI (for minimization)
            if sigma > 1e-10:
                z = (best_error - mu - self.xi) / sigma
                ei = (best_error - mu - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)
                ei = max(0, ei)
            else:
                ei = 0.0 if mu > best_error else (best_error - mu)

            scores.append(ei)

        return np.array(scores)


class QueryByCommittee(AcquisitionFunction):
    """
    Query-by-Committee (QBC) acquisition function.

    Measures disagreement among ensemble members. Points where models
    disagree most are the most informative to label.

    Disagreement can be measured as:
    - Variance of predictions across committee
    - Vote entropy (for classification)
    - KL divergence between predictions
    """

    def __init__(self, disagreement_metric: str = "variance"):
        """
        Initialize QBC acquisition.

        Args:
            disagreement_metric: How to measure disagreement
                                ("variance", "range", "cv")
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
            if self.disagreement_metric == "variance":
                disagreement = np.mean(np.var(predictions, axis=0))
            elif self.disagreement_metric == "range":
                disagreement = np.mean(np.ptp(predictions, axis=0))
            elif self.disagreement_metric == "cv":
                # Coefficient of variation
                mean_pred = np.mean(predictions, axis=0)
                std_pred = np.std(predictions, axis=0)
                cv = std_pred / (np.abs(mean_pred) + 1e-10)
                disagreement = np.mean(cv)
            else:
                disagreement = np.mean(np.var(predictions, axis=0))

            scores.append(disagreement)

        return np.array(scores)


class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper Confidence Bound (UCB) acquisition function.

    Optimistic strategy that selects points with high predicted error
    plus uncertainty bonus. Common in bandit algorithms.

    UCB(x) = mu(x) + kappa * sigma(x)

    For error maximization (finding high error regions):
    Higher UCB = potentially high error region
    """

    def __init__(self, kappa: float = 2.0):
        """
        Initialize UCB acquisition.

        Args:
            kappa: Exploration parameter. Higher = more exploration.
                  Typical values: 1.0-3.0
        """
        super().__init__(name="ucb")
        self.kappa = kappa

    def compute(
        self,
        candidates: np.ndarray,
        model: SurrogateModel,
        coordinates: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Compute UCB scores."""
        scores = []

        for i in range(len(candidates)):
            params = candidates[i:i+1]
            result = model.predict(params, coordinates)

            # Mean prediction magnitude (proxy for expected error)
            mu = np.mean(np.abs(result.values))

            # Uncertainty
            if result.uncertainty is not None:
                sigma = np.mean(result.uncertainty)
            else:
                sigma = 0.0

            # UCB score
            ucb = mu + self.kappa * sigma
            scores.append(ucb)

        return np.array(scores)


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Probability of Improvement (PI) acquisition function.

    Computes the probability that a candidate will improve upon
    the current best observation.

    PI(x) = P(f(x) > f_best + xi) = Phi((mu(x) - f_best - xi) / sigma(x))
    """

    def __init__(self, xi: float = 0.01):
        """
        Initialize PI acquisition.

        Args:
            xi: Improvement threshold
        """
        super().__init__(name="probability_of_improvement")
        self.xi = xi

    def compute(
        self,
        candidates: np.ndarray,
        model: SurrogateModel,
        coordinates: np.ndarray,
        best_error: float = 0.0,
        **kwargs
    ) -> np.ndarray:
        """Compute PI scores."""
        from scipy.stats import norm

        scores = []

        for i in range(len(candidates)):
            params = candidates[i:i+1]
            result = model.predict(params, coordinates)

            mu = np.mean(np.abs(result.values))

            if result.uncertainty is not None:
                sigma = np.mean(result.uncertainty)
            else:
                sigma = 0.1

            # PI for finding high-error regions
            if sigma > 1e-10:
                z = (mu - best_error - self.xi) / sigma
                pi = norm.cdf(z)
            else:
                pi = 1.0 if mu > best_error + self.xi else 0.0

            scores.append(pi)

        return np.array(scores)


class HybridAcquisition(AcquisitionFunction):
    """
    Hybrid acquisition combining multiple strategies.

    Useful for balancing different objectives:
    - Uncertainty (exploration)
    - Expected improvement (exploitation)
    - Diversity (coverage)
    """

    def __init__(
        self,
        strategies: List[Tuple[AcquisitionFunction, float]],
        normalize: bool = True
    ):
        """
        Initialize hybrid acquisition.

        Args:
            strategies: List of (acquisition_fn, weight) tuples
            normalize: Whether to normalize scores before combining
        """
        super().__init__(name="hybrid")
        self.strategies = strategies
        self.normalize = normalize

    def compute(
        self,
        candidates: np.ndarray,
        model: SurrogateModel,
        coordinates: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Compute weighted combination of acquisition scores."""
        combined_scores = np.zeros(len(candidates))

        for acquisition_fn, weight in self.strategies:
            scores = acquisition_fn.compute(candidates, model, coordinates, **kwargs)

            if self.normalize and scores.std() > 1e-10:
                scores = (scores - scores.mean()) / scores.std()

            combined_scores += weight * scores

        return combined_scores


class GradientBasedAcquisition(AcquisitionFunction):
    """
    Acquisition based on output sensitivity to parameter changes.

    Regions where small parameter changes cause large output changes
    need more samples for accurate interpolation.
    """

    def __init__(self, epsilon: float = 0.01):
        """
        Initialize gradient-based acquisition.

        Args:
            epsilon: Perturbation size for finite differences
        """
        super().__init__(name="gradient")
        self.epsilon = epsilon

    def compute(
        self,
        candidates: np.ndarray,
        model: SurrogateModel,
        coordinates: np.ndarray,
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        **kwargs
    ) -> np.ndarray:
        """Compute gradient-based sensitivity scores."""
        scores = []
        n_params = candidates.shape[1]

        for i in range(len(candidates)):
            params = candidates[i:i+1]
            result_center = model.predict(params, coordinates)
            center_vals = result_center.values.flatten()

            gradients = []
            for p in range(n_params):
                # Perturb parameter
                params_plus = params.copy()
                params_plus[0, p] += self.epsilon

                params_minus = params.copy()
                params_minus[0, p] -= self.epsilon

                # Predictions at perturbed points
                result_plus = model.predict(params_plus, coordinates)
                result_minus = model.predict(params_minus, coordinates)

                # Finite difference gradient
                grad = (result_plus.values.flatten() - result_minus.values.flatten()) / (2 * self.epsilon)
                gradients.append(np.mean(np.abs(grad)))

            # Total sensitivity
            sensitivity = np.sum(gradients)
            scores.append(sensitivity)

        return np.array(scores)


def get_acquisition_function(
    acquisition_type: Union[str, AcquisitionType],
    **kwargs
) -> AcquisitionFunction:
    """
    Factory function to create acquisition functions.

    Args:
        acquisition_type: Type of acquisition function
        **kwargs: Arguments for the specific acquisition function

    Returns:
        AcquisitionFunction instance
    """
    if isinstance(acquisition_type, str):
        acquisition_type = AcquisitionType[acquisition_type.upper()]

    if acquisition_type == AcquisitionType.UNCERTAINTY:
        return UncertaintySampling(**kwargs)
    elif acquisition_type == AcquisitionType.EXPECTED_IMPROVEMENT:
        return ExpectedImprovement(**kwargs)
    elif acquisition_type == AcquisitionType.QUERY_BY_COMMITTEE:
        return QueryByCommittee(**kwargs)
    elif acquisition_type == AcquisitionType.UCB:
        return UpperConfidenceBound(**kwargs)
    elif acquisition_type == AcquisitionType.PROBABILITY_OF_IMPROVEMENT:
        return ProbabilityOfImprovement(**kwargs)
    elif acquisition_type == AcquisitionType.GREEDY:
        return UncertaintySampling(**kwargs)  # Default to uncertainty
    elif acquisition_type == AcquisitionType.HYBRID:
        # Create default hybrid
        strategies = [
            (UncertaintySampling(), 0.5),
            (ExpectedImprovement(), 0.5),
        ]
        return HybridAcquisition(strategies, **kwargs)
    else:
        raise ValueError(f"Unknown acquisition type: {acquisition_type}")
