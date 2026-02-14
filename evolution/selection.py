"""
Parent selection strategies for evolutionary search.

Implements various selection methods for choosing parents
for the next generation of solutions.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from evolution.solution import Solution


@dataclass
class SelectionResult:
    """Result of parent selection."""
    selected: List[Solution]
    scores: Dict[str, float]  # solution_id -> score
    method: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": [s.id for s in self.selected],
            "scores": self.scores,
            "method": self.method,
            "metadata": self.metadata,
        }


class ParentSelector(ABC):
    """Abstract base class for parent selection strategies."""

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Name of the selection method."""
        pass

    @abstractmethod
    def select(
        self,
        candidates: List[Solution],
        n: int,
        **kwargs,
    ) -> SelectionResult:
        """
        Select n parents from candidates.

        Args:
            candidates: List of candidate solutions
            n: Number of parents to select
            **kwargs: Additional parameters

        Returns:
            SelectionResult with selected parents
        """
        pass


class TournamentSelector(ParentSelector):
    """
    Tournament selection.

    Randomly samples k candidates and selects the best one.
    Repeat to get n parents.
    """

    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selector.

        Args:
            tournament_size: Number of candidates per tournament
        """
        self.tournament_size = tournament_size

    @property
    def method_name(self) -> str:
        return "tournament"

    def select(
        self,
        candidates: List[Solution],
        n: int,
        **kwargs,
    ) -> SelectionResult:
        """Select using tournament selection."""
        if not candidates:
            return SelectionResult(
                selected=[],
                scores={},
                method=self.method_name,
                metadata={"tournament_size": self.tournament_size},
            )

        if len(candidates) <= n:
            scores = {s.id: s.metrics.compute_overall_score() for s in candidates}
            return SelectionResult(
                selected=candidates,
                scores=scores,
                method=self.method_name,
                metadata={"tournament_size": self.tournament_size},
            )

        selected = []
        scores = {}

        for _ in range(n):
            # Sample tournament
            tournament_size = min(self.tournament_size, len(candidates))
            tournament = random.sample(candidates, tournament_size)

            # Select winner
            winner = max(
                tournament,
                key=lambda s: s.metrics.compute_overall_score()
            )

            selected.append(winner)
            scores[winner.id] = winner.metrics.compute_overall_score()

        return SelectionResult(
            selected=selected,
            scores=scores,
            method=self.method_name,
            metadata={"tournament_size": self.tournament_size},
        )


class RouletteSelector(ParentSelector):
    """
    Roulette wheel (fitness proportionate) selection.

    Selection probability proportional to fitness score.
    """

    @property
    def method_name(self) -> str:
        return "roulette"

    def select(
        self,
        candidates: List[Solution],
        n: int,
        **kwargs,
    ) -> SelectionResult:
        """Select using roulette wheel selection."""
        if not candidates:
            return SelectionResult(
                selected=[],
                scores={},
                method=self.method_name,
                metadata={},
            )

        # Calculate fitness scores
        scores = {s.id: s.metrics.compute_overall_score() for s in candidates}

        # Shift to ensure all positive
        min_score = min(scores.values())
        shifted_scores = {sid: score - min_score + 0.01 for sid, score in scores.items()}

        # Calculate probabilities
        total = sum(shifted_scores.values())
        probabilities = {sid: score / total for sid, score in shifted_scores.items()}

        # Select n parents
        selected = []
        candidate_ids = list(probabilities.keys())
        probs = [probabilities[cid] for cid in candidate_ids]
        id_to_solution = {s.id: s for s in candidates}

        for _ in range(min(n, len(candidates))):
            chosen_id = random.choices(candidate_ids, weights=probs, k=1)[0]
            selected.append(id_to_solution[chosen_id])

        return SelectionResult(
            selected=selected,
            scores=scores,
            method=self.method_name,
            metadata={"probabilities": probabilities},
        )


class ElitistSelector(ParentSelector):
    """
    Elitist selection.

    Simply selects the top n solutions by score.
    """

    @property
    def method_name(self) -> str:
        return "elitist"

    def select(
        self,
        candidates: List[Solution],
        n: int,
        **kwargs,
    ) -> SelectionResult:
        """Select top n by score."""
        if not candidates:
            return SelectionResult(
                selected=[],
                scores={},
                method=self.method_name,
                metadata={},
            )

        scores = {s.id: s.metrics.compute_overall_score() for s in candidates}

        sorted_candidates = sorted(
            candidates,
            key=lambda s: scores[s.id],
            reverse=True
        )

        selected = sorted_candidates[:n]

        return SelectionResult(
            selected=selected,
            scores=scores,
            method=self.method_name,
            metadata={},
        )


class DiversitySelector(ParentSelector):
    """
    Diversity-aware selection.

    Balances quality with diversity to maintain exploration.
    """

    def __init__(self, diversity_weight: float = 0.3):
        """
        Initialize diversity selector.

        Args:
            diversity_weight: Weight for diversity (0-1)
        """
        self.diversity_weight = diversity_weight

    @property
    def method_name(self) -> str:
        return "diversity"

    def select(
        self,
        candidates: List[Solution],
        n: int,
        **kwargs,
    ) -> SelectionResult:
        """Select balancing quality and diversity."""
        if not candidates:
            return SelectionResult(
                selected=[],
                scores={},
                method=self.method_name,
                metadata={"diversity_weight": self.diversity_weight},
            )

        if len(candidates) <= n:
            scores = {s.id: s.metrics.compute_overall_score() for s in candidates}
            return SelectionResult(
                selected=candidates,
                scores=scores,
                method=self.method_name,
                metadata={"diversity_weight": self.diversity_weight},
            )

        # Calculate base scores
        base_scores = {s.id: s.metrics.compute_overall_score() for s in candidates}

        selected = []
        remaining = list(candidates)

        # Select first by pure quality
        first = max(remaining, key=lambda s: base_scores[s.id])
        selected.append(first)
        remaining.remove(first)

        # Select rest balancing quality and diversity
        while len(selected) < n and remaining:
            best_candidate = None
            best_combined_score = -1

            for candidate in remaining:
                quality_score = base_scores[candidate.id]

                # Calculate diversity from already selected
                diversity_score = self._calculate_diversity_from_selected(
                    candidate, selected
                )

                combined = (
                    (1 - self.diversity_weight) * quality_score +
                    self.diversity_weight * diversity_score
                )

                if combined > best_combined_score:
                    best_combined_score = combined
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return SelectionResult(
            selected=selected,
            scores=base_scores,
            method=self.method_name,
            metadata={"diversity_weight": self.diversity_weight},
        )

    def _calculate_diversity_from_selected(
        self,
        candidate: Solution,
        selected: List[Solution],
    ) -> float:
        """Calculate average distance from candidate to selected solutions."""
        if not selected:
            return 1.0

        distances = [
            candidate.genome.distance(s.genome)
            for s in selected
        ]

        # Normalize (assuming max meaningful distance is ~2)
        avg_distance = sum(distances) / len(distances)
        return min(1.0, avg_distance / 2.0)


class EnsembleSelector(ParentSelector):
    """
    Ensemble selection using multiple strategies.

    Combines votes from multiple selection methods.
    """

    def __init__(
        self,
        selectors: Optional[List[Tuple[ParentSelector, float]]] = None,
    ):
        """
        Initialize ensemble selector.

        Args:
            selectors: List of (selector, weight) tuples
        """
        if selectors is None:
            selectors = [
                (TournamentSelector(3), 0.4),
                (ElitistSelector(), 0.3),
                (DiversitySelector(0.3), 0.3),
            ]
        self.selectors = selectors

    @property
    def method_name(self) -> str:
        return "ensemble"

    def select(
        self,
        candidates: List[Solution],
        n: int,
        **kwargs,
    ) -> SelectionResult:
        """Select using ensemble of methods."""
        if not candidates:
            return SelectionResult(
                selected=[],
                scores={},
                method=self.method_name,
                metadata={},
            )

        # Collect votes from each selector
        votes: Dict[str, float] = {s.id: 0.0 for s in candidates}

        for selector, weight in self.selectors:
            result = selector.select(candidates, n)

            # Weight votes by position
            for i, solution in enumerate(result.selected):
                position_weight = 1.0 - (i / n) * 0.5  # First gets 1.0, last gets 0.5
                votes[solution.id] += weight * position_weight

        # Normalize votes
        max_vote = max(votes.values()) if votes else 1.0
        normalized_votes = {sid: v / max_vote for sid, v in votes.items()}

        # Select top n by votes
        sorted_candidates = sorted(
            candidates,
            key=lambda s: votes[s.id],
            reverse=True
        )

        selected = sorted_candidates[:n]

        return SelectionResult(
            selected=selected,
            scores={s.id: s.metrics.compute_overall_score() for s in candidates},
            method=self.method_name,
            metadata={
                "votes": normalized_votes,
                "component_methods": [s.method_name for s, _ in self.selectors],
            },
        )


def create_selector(method: str, **kwargs) -> ParentSelector:
    """
    Factory function to create a selector.

    Args:
        method: Selection method name
        **kwargs: Method-specific parameters

    Returns:
        ParentSelector instance
    """
    if method == "tournament":
        return TournamentSelector(kwargs.get("tournament_size", 3))
    elif method == "roulette":
        return RouletteSelector()
    elif method == "elitist":
        return ElitistSelector()
    elif method == "diversity":
        return DiversitySelector(kwargs.get("diversity_weight", 0.3))
    elif method == "ensemble":
        return EnsembleSelector()
    else:
        raise ValueError(f"Unknown selection method: {method}")
