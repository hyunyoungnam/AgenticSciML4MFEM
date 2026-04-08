"""
Active learning metrics and efficiency tracking.

Provides tools to monitor the performance of the active learning loop,
track sample efficiency, and determine optimal stopping points.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np


@dataclass
class IterationMetrics:
    """
    Metrics for a single active learning iteration.

    Attributes:
        iteration: Iteration number
        n_samples_total: Total samples after this iteration
        n_samples_new: New samples added in this iteration
        train_error: Training error
        test_error: Test/validation error
        mean_uncertainty: Mean model uncertainty
        max_uncertainty: Maximum model uncertainty
        n_weak_regions: Number of weak regions identified
        acquisition_scores: Acquisition scores for selected samples
        timestamp: When this iteration completed
    """
    iteration: int
    n_samples_total: int
    n_samples_new: int
    train_error: float
    test_error: float
    mean_uncertainty: float = 0.0
    max_uncertainty: float = 0.0
    n_weak_regions: int = 0
    acquisition_scores: List[float] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "n_samples_total": self.n_samples_total,
            "n_samples_new": self.n_samples_new,
            "train_error": self.train_error,
            "test_error": self.test_error,
            "mean_uncertainty": self.mean_uncertainty,
            "max_uncertainty": self.max_uncertainty,
            "n_weak_regions": self.n_weak_regions,
            "acquisition_scores": self.acquisition_scores,
            "timestamp": self.timestamp,
        }


@dataclass
class ActiveLearningStats:
    """
    Aggregated statistics for the active learning process.

    Attributes:
        total_iterations: Total iterations completed
        total_samples: Total samples generated
        final_error: Final test error achieved
        error_reduction: Total error reduction from initial
        sample_efficiency: Error reduction per sample
        convergence_rate: Rate of error decrease
        stopping_reason: Why the loop stopped
    """
    total_iterations: int
    total_samples: int
    final_error: float
    initial_error: float
    error_reduction: float
    sample_efficiency: float
    convergence_rate: float
    stopping_reason: str
    diminishing_returns_detected: bool = False
    optimal_stop_iteration: Optional[int] = None


class ActiveLearningMetrics:
    """
    Tracks and analyzes active learning performance.

    Provides:
    1. Per-iteration metrics logging
    2. Sample efficiency analysis
    3. Convergence detection
    4. Stopping criteria evaluation
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.history: List[IterationMetrics] = []
        self._start_time: Optional[str] = None

    def log_iteration(
        self,
        iteration: int,
        n_samples_total: int,
        n_samples_new: int,
        train_error: float,
        test_error: float,
        mean_uncertainty: float = 0.0,
        max_uncertainty: float = 0.0,
        n_weak_regions: int = 0,
        acquisition_scores: Optional[List[float]] = None
    ) -> IterationMetrics:
        """
        Log metrics for a completed iteration.

        Args:
            iteration: Current iteration number
            n_samples_total: Total samples in dataset
            n_samples_new: New samples added this iteration
            train_error: Training error
            test_error: Validation/test error
            mean_uncertainty: Mean model uncertainty
            max_uncertainty: Maximum uncertainty
            n_weak_regions: Number of weak regions detected
            acquisition_scores: Acquisition scores for selected samples

        Returns:
            IterationMetrics object
        """
        if self._start_time is None:
            self._start_time = datetime.now().isoformat()

        metrics = IterationMetrics(
            iteration=iteration,
            n_samples_total=n_samples_total,
            n_samples_new=n_samples_new,
            train_error=train_error,
            test_error=test_error,
            mean_uncertainty=mean_uncertainty,
            max_uncertainty=max_uncertainty,
            n_weak_regions=n_weak_regions,
            acquisition_scores=acquisition_scores or [],
        )

        self.history.append(metrics)
        return metrics

    def compute_efficiency(self) -> float:
        """
        Compute overall sample efficiency.

        Returns error reduction per sample added.
        """
        if len(self.history) < 2:
            return 0.0

        initial_error = self.history[0].test_error
        final_error = self.history[-1].test_error
        initial_samples = self.history[0].n_samples_total
        final_samples = self.history[-1].n_samples_total

        samples_added = final_samples - initial_samples
        if samples_added <= 0:
            return 0.0

        error_reduction = initial_error - final_error
        return error_reduction / samples_added

    def compute_iteration_efficiency(self) -> List[float]:
        """
        Compute per-iteration sample efficiency.

        Returns list of error reduction per sample for each iteration.
        """
        efficiencies = []

        for i in range(1, len(self.history)):
            prev = self.history[i - 1]
            curr = self.history[i]

            samples_added = curr.n_samples_total - prev.n_samples_total
            error_reduction = prev.test_error - curr.test_error

            if samples_added > 0:
                efficiency = error_reduction / samples_added
            else:
                efficiency = 0.0

            efficiencies.append(efficiency)

        return efficiencies

    def compute_convergence_rate(self, window: int = 3) -> float:
        """
        Compute recent convergence rate.

        Args:
            window: Number of recent iterations to consider

        Returns:
            Average error reduction rate over window
        """
        if len(self.history) < window + 1:
            return float('inf')

        recent = self.history[-window:]
        errors = [m.test_error for m in recent]

        # Linear fit to estimate rate
        x = np.arange(len(errors))
        if len(x) > 1:
            slope = np.polyfit(x, errors, 1)[0]
            return -slope  # Negative slope = decreasing error
        return 0.0

    def detect_diminishing_returns(
        self,
        window: int = 3,
        threshold: float = 0.01
    ) -> bool:
        """
        Detect if error reduction has plateaued.

        Args:
            window: Number of iterations to analyze
            threshold: Minimum improvement to not be considered diminishing

        Returns:
            True if diminishing returns detected
        """
        if len(self.history) < window + 1:
            return False

        recent_errors = [m.test_error for m in self.history[-window-1:]]
        improvements = [recent_errors[i] - recent_errors[i+1] for i in range(len(recent_errors)-1)]

        # Check if all recent improvements are below threshold
        return all(imp < threshold for imp in improvements)

    def suggest_termination(
        self,
        min_improvement: float = 0.001,
        patience: int = 3,
        max_samples: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Suggest whether to terminate the active learning loop.

        Args:
            min_improvement: Minimum error improvement per iteration
            patience: Number of iterations without improvement before stopping
            max_samples: Maximum sample budget

        Returns:
            Tuple of (should_stop, reason)
        """
        if not self.history:
            return False, "No iterations completed"

        current = self.history[-1]

        # Check sample budget
        if max_samples and current.n_samples_total >= max_samples:
            return True, f"Sample budget exhausted ({max_samples})"

        # Check for plateau
        if self.detect_diminishing_returns(window=patience, threshold=min_improvement):
            return True, f"Diminishing returns: no improvement > {min_improvement} for {patience} iterations"

        # Check if uncertainty is low
        if current.mean_uncertainty < min_improvement * 0.1:
            return True, "Model uncertainty is very low"

        # Check if no weak regions remain
        if current.n_weak_regions == 0:
            return True, "No weak regions identified"

        return False, "Continue learning"

    def find_optimal_stopping_point(self) -> Optional[int]:
        """
        Find the optimal stopping point based on efficiency.

        Uses the "elbow" method to find where adding more samples
        yields diminishing returns.

        Returns:
            Optimal iteration number, or None if not enough data
        """
        if len(self.history) < 4:
            return None

        # Compute cumulative efficiency curve
        samples = [m.n_samples_total for m in self.history]
        errors = [m.test_error for m in self.history]

        # Normalize
        samples = np.array(samples)
        errors = np.array(errors)

        if samples.max() > samples.min():
            samples_norm = (samples - samples.min()) / (samples.max() - samples.min())
        else:
            samples_norm = samples

        if errors.max() > errors.min():
            errors_norm = (errors - errors.min()) / (errors.max() - errors.min())
        else:
            errors_norm = errors

        # Find elbow using distance to line method
        # Line from first to last point
        p1 = np.array([samples_norm[0], errors_norm[0]])
        p2 = np.array([samples_norm[-1], errors_norm[-1]])

        distances = []
        for i in range(len(samples)):
            p = np.array([samples_norm[i], errors_norm[i]])
            # Distance from point to line
            d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
            distances.append(d)

        # Elbow is point with maximum distance
        elbow_idx = np.argmax(distances)

        return self.history[elbow_idx].iteration

    def get_statistics(self) -> ActiveLearningStats:
        """
        Compute aggregate statistics for the learning process.

        Returns:
            ActiveLearningStats with all aggregate metrics
        """
        if not self.history:
            return ActiveLearningStats(
                total_iterations=0,
                total_samples=0,
                final_error=float('inf'),
                initial_error=float('inf'),
                error_reduction=0.0,
                sample_efficiency=0.0,
                convergence_rate=0.0,
                stopping_reason="No iterations",
            )

        initial = self.history[0]
        final = self.history[-1]

        error_reduction = initial.test_error - final.test_error
        sample_efficiency = self.compute_efficiency()
        convergence_rate = self.compute_convergence_rate()
        diminishing = self.detect_diminishing_returns()
        optimal_stop = self.find_optimal_stopping_point()

        # Determine stopping reason
        should_stop, reason = self.suggest_termination()

        return ActiveLearningStats(
            total_iterations=len(self.history),
            total_samples=final.n_samples_total,
            final_error=final.test_error,
            initial_error=initial.test_error,
            error_reduction=error_reduction,
            sample_efficiency=sample_efficiency,
            convergence_rate=convergence_rate,
            stopping_reason=reason,
            diminishing_returns_detected=diminishing,
            optimal_stop_iteration=optimal_stop,
        )

    def get_error_history(self) -> Tuple[List[int], List[float], List[float]]:
        """
        Get error history for plotting.

        Returns:
            Tuple of (sample_counts, train_errors, test_errors)
        """
        samples = [m.n_samples_total for m in self.history]
        train = [m.train_error for m in self.history]
        test = [m.test_error for m in self.history]
        return samples, train, test

    def get_uncertainty_history(self) -> Tuple[List[int], List[float], List[float]]:
        """
        Get uncertainty history for plotting.

        Returns:
            Tuple of (sample_counts, mean_uncertainties, max_uncertainties)
        """
        samples = [m.n_samples_total for m in self.history]
        mean_unc = [m.mean_uncertainty for m in self.history]
        max_unc = [m.max_uncertainty for m in self.history]
        return samples, mean_unc, max_unc

    def save(self, path: Path) -> None:
        """
        Save metrics to JSON file.

        Args:
            path: Path to save file
        """
        data = {
            "start_time": self._start_time,
            "history": [m.to_dict() for m in self.history],
            "statistics": {
                "efficiency": self.compute_efficiency(),
                "convergence_rate": self.compute_convergence_rate(),
                "diminishing_returns": self.detect_diminishing_returns(),
            }
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ActiveLearningMetrics":
        """
        Load metrics from JSON file.

        Args:
            path: Path to load from

        Returns:
            ActiveLearningMetrics instance
        """
        with open(path, 'r') as f:
            data = json.load(f)

        metrics = cls()
        metrics._start_time = data.get("start_time")

        for item in data.get("history", []):
            metrics.history.append(IterationMetrics(
                iteration=item["iteration"],
                n_samples_total=item["n_samples_total"],
                n_samples_new=item["n_samples_new"],
                train_error=item["train_error"],
                test_error=item["test_error"],
                mean_uncertainty=item.get("mean_uncertainty", 0.0),
                max_uncertainty=item.get("max_uncertainty", 0.0),
                n_weak_regions=item.get("n_weak_regions", 0),
                acquisition_scores=item.get("acquisition_scores", []),
                timestamp=item.get("timestamp", ""),
            ))

        return metrics

    def summary(self) -> str:
        """
        Generate human-readable summary.

        Returns:
            Summary string
        """
        if not self.history:
            return "No active learning history recorded."

        stats = self.get_statistics()

        lines = [
            "=== Active Learning Summary ===",
            f"Total iterations: {stats.total_iterations}",
            f"Total samples: {stats.total_samples}",
            f"Initial error: {stats.initial_error:.6f}",
            f"Final error: {stats.final_error:.6f}",
            f"Error reduction: {stats.error_reduction:.6f} ({100*stats.error_reduction/stats.initial_error:.1f}%)",
            f"Sample efficiency: {stats.sample_efficiency:.6f} (error reduction per sample)",
            f"Convergence rate: {stats.convergence_rate:.6f}",
            f"Diminishing returns: {'Yes' if stats.diminishing_returns_detected else 'No'}",
        ]

        if stats.optimal_stop_iteration:
            lines.append(f"Optimal stop point: iteration {stats.optimal_stop_iteration}")

        lines.append(f"Status: {stats.stopping_reason}")

        return "\n".join(lines)


class ConvergenceMonitor:
    """
    Monitors convergence of the active learning process.

    Provides real-time convergence analysis and early stopping
    recommendations.
    """

    def __init__(
        self,
        target_error: float = 0.01,
        patience: int = 3,
        min_improvement: float = 0.001,
        max_samples: Optional[int] = None
    ):
        """
        Initialize convergence monitor.

        Args:
            target_error: Target error threshold for convergence
            patience: Iterations without improvement before stopping
            min_improvement: Minimum improvement to reset patience
            max_samples: Maximum sample budget
        """
        self.target_error = target_error
        self.patience = patience
        self.min_improvement = min_improvement
        self.max_samples = max_samples

        self._best_error = float('inf')
        self._patience_counter = 0
        self._converged = False
        self._stop_reason: Optional[str] = None

    def update(
        self,
        test_error: float,
        n_samples: int
    ) -> Tuple[bool, str]:
        """
        Update monitor with new iteration results.

        Args:
            test_error: Current test error
            n_samples: Current sample count

        Returns:
            Tuple of (should_stop, reason)
        """
        # Check target error
        if test_error <= self.target_error:
            self._converged = True
            self._stop_reason = f"Target error reached ({test_error:.6f} <= {self.target_error})"
            return True, self._stop_reason

        # Check sample budget
        if self.max_samples and n_samples >= self.max_samples:
            self._stop_reason = f"Sample budget exhausted ({n_samples} >= {self.max_samples})"
            return True, self._stop_reason

        # Check improvement
        improvement = self._best_error - test_error
        if improvement > self.min_improvement:
            self._best_error = test_error
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        # Check patience
        if self._patience_counter >= self.patience:
            self._stop_reason = f"No improvement for {self.patience} iterations"
            return True, self._stop_reason

        return False, "Continue"

    def reset(self):
        """Reset monitor state."""
        self._best_error = float('inf')
        self._patience_counter = 0
        self._converged = False
        self._stop_reason = None

    @property
    def is_converged(self) -> bool:
        """Check if converged to target."""
        return self._converged

    @property
    def best_error(self) -> float:
        """Get best error seen."""
        return self._best_error

    @property
    def remaining_patience(self) -> int:
        """Get remaining patience."""
        return max(0, self.patience - self._patience_counter)
