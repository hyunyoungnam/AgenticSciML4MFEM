"""
Solution tree manager for evolutionary search.

Maintains the tree structure of solutions across generations,
supporting parent-child relationships and lineage tracking.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from inpforge.evolution.solution import Solution, SolutionStatus, SolutionGenome, SolutionMetrics


@dataclass
class GenerationStats:
    """Statistics for a generation."""
    generation: int
    num_solutions: int
    num_converged: int
    num_failed: int
    avg_score: float
    best_score: float
    best_solution_id: Optional[str]
    diversity: float  # Genome diversity metric

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "num_solutions": self.num_solutions,
            "num_converged": self.num_converged,
            "num_failed": self.num_failed,
            "avg_score": self.avg_score,
            "best_score": self.best_score,
            "best_solution_id": self.best_solution_id,
            "diversity": self.diversity,
        }


class SolutionTree:
    """
    Tree structure for managing solutions across generations.

    Maintains parent-child relationships and provides methods
    for traversal, selection, and analysis.
    """

    def __init__(self, max_generations: int = 20):
        """
        Initialize the solution tree.

        Args:
            max_generations: Maximum number of generations
        """
        self.max_generations = max_generations
        self.solutions: Dict[str, Solution] = {}
        self.root_id: Optional[str] = None
        self.current_generation: int = 0
        self._children: Dict[str, Set[str]] = {}  # parent_id -> set of child_ids
        self._by_generation: Dict[int, Set[str]] = {}  # generation -> set of solution_ids

    def add_solution(self, solution: Solution) -> None:
        """
        Add a solution to the tree.

        Args:
            solution: Solution to add
        """
        self.solutions[solution.id] = solution

        # Track parent-child relationship
        if solution.parent_id:
            if solution.parent_id not in self._children:
                self._children[solution.parent_id] = set()
            self._children[solution.parent_id].add(solution.id)
        else:
            # This is a root solution
            if self.root_id is None:
                self.root_id = solution.id

        # Track by generation
        gen = solution.generation
        if gen not in self._by_generation:
            self._by_generation[gen] = set()
        self._by_generation[gen].add(solution.id)

        # Update current generation
        self.current_generation = max(self.current_generation, gen)

    def get_solution(self, solution_id: str) -> Optional[Solution]:
        """Get a solution by ID."""
        return self.solutions.get(solution_id)

    def get_children(self, solution_id: str) -> List[Solution]:
        """Get all children of a solution."""
        child_ids = self._children.get(solution_id, set())
        return [self.solutions[cid] for cid in child_ids if cid in self.solutions]

    def get_parent(self, solution_id: str) -> Optional[Solution]:
        """Get the parent of a solution."""
        solution = self.solutions.get(solution_id)
        if solution and solution.parent_id:
            return self.solutions.get(solution.parent_id)
        return None

    def get_generation(self, generation: int) -> List[Solution]:
        """Get all solutions in a generation."""
        solution_ids = self._by_generation.get(generation, set())
        return [self.solutions[sid] for sid in solution_ids if sid in self.solutions]

    def get_current_generation(self) -> List[Solution]:
        """Get all solutions in the current generation."""
        return self.get_generation(self.current_generation)

    def get_lineage(self, solution_id: str) -> List[Solution]:
        """
        Get the lineage from root to the specified solution.

        Args:
            solution_id: Solution ID

        Returns:
            List of solutions from root to this solution
        """
        lineage = []
        current = self.solutions.get(solution_id)

        while current:
            lineage.append(current)
            if current.parent_id:
                current = self.solutions.get(current.parent_id)
            else:
                break

        lineage.reverse()
        return lineage

    def get_converged_solutions(self) -> List[Solution]:
        """Get all converged solutions."""
        return [
            s for s in self.solutions.values()
            if s.status == SolutionStatus.CONVERGED
        ]

    def get_failed_solutions(self) -> List[Solution]:
        """Get all failed solutions."""
        return [
            s for s in self.solutions.values()
            if s.status == SolutionStatus.FAILED
        ]

    def get_best_solutions(self, n: int = 5, generation: Optional[int] = None) -> List[Solution]:
        """
        Get the top N solutions by score.

        Args:
            n: Number of solutions to return
            generation: Optional generation filter

        Returns:
            List of top solutions
        """
        if generation is not None:
            candidates = self.get_generation(generation)
        else:
            candidates = list(self.solutions.values())

        # Filter to converged/executed solutions
        candidates = [
            s for s in candidates
            if s.status in (SolutionStatus.CONVERGED, SolutionStatus.EXECUTED)
        ]

        # Sort by metrics score
        candidates.sort(
            key=lambda s: s.metrics.compute_overall_score(),
            reverse=True
        )

        return candidates[:n]

    def get_generation_stats(self, generation: int) -> GenerationStats:
        """Get statistics for a generation."""
        solutions = self.get_generation(generation)

        if not solutions:
            return GenerationStats(
                generation=generation,
                num_solutions=0,
                num_converged=0,
                num_failed=0,
                avg_score=0.0,
                best_score=0.0,
                best_solution_id=None,
                diversity=0.0,
            )

        scores = [s.metrics.compute_overall_score() for s in solutions]
        converged = [s for s in solutions if s.status == SolutionStatus.CONVERGED]
        failed = [s for s in solutions if s.status == SolutionStatus.FAILED]

        best_idx = scores.index(max(scores)) if scores else 0
        best_solution = solutions[best_idx] if solutions else None

        # Calculate diversity (average pairwise genome distance)
        diversity = self._calculate_diversity(solutions)

        return GenerationStats(
            generation=generation,
            num_solutions=len(solutions),
            num_converged=len(converged),
            num_failed=len(failed),
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            best_score=max(scores) if scores else 0.0,
            best_solution_id=best_solution.id if best_solution else None,
            diversity=diversity,
        )

    def _calculate_diversity(self, solutions: List[Solution]) -> float:
        """Calculate genome diversity for a set of solutions."""
        if len(solutions) < 2:
            return 0.0

        total_distance = 0.0
        pairs = 0

        for i, s1 in enumerate(solutions):
            for s2 in solutions[i + 1:]:
                total_distance += s1.genome.distance(s2.genome)
                pairs += 1

        return total_distance / pairs if pairs > 0 else 0.0

    def get_eligible_parents(
        self,
        generation: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[Solution]:
        """
        Get solutions eligible to be parents for the next generation.

        Args:
            generation: Generation to select from (default: current)
            min_score: Minimum score threshold

        Returns:
            List of eligible parent solutions
        """
        if generation is None:
            generation = self.current_generation

        solutions = self.get_generation(generation)

        eligible = [
            s for s in solutions
            if s.is_successful() and s.metrics.compute_overall_score() >= min_score
        ]

        return eligible

    def create_child(
        self,
        parent_id: str,
        genome: SolutionGenome,
    ) -> Solution:
        """
        Create a new child solution.

        Args:
            parent_id: Parent solution ID
            genome: Genome for the new solution

        Returns:
            New Solution instance (not yet added to tree)
        """
        parent = self.solutions.get(parent_id)
        if not parent:
            raise ValueError(f"Parent solution not found: {parent_id}")

        child = Solution(
            parent_id=parent_id,
            generation=parent.generation + 1,
            genome=genome,
            status=SolutionStatus.PROPOSED,
        )

        return child

    def prune_failed(self, keep_latest: int = 10) -> int:
        """
        Prune failed solutions to save memory.

        Args:
            keep_latest: Number of recent failed solutions to keep

        Returns:
            Number of solutions pruned
        """
        failed = sorted(
            self.get_failed_solutions(),
            key=lambda s: s.created_at,
            reverse=True
        )

        to_prune = failed[keep_latest:]
        pruned = 0

        for solution in to_prune:
            # Don't prune if it has children
            if solution.id in self._children and self._children[solution.id]:
                continue

            # Remove from tree
            del self.solutions[solution.id]
            if solution.parent_id and solution.parent_id in self._children:
                self._children[solution.parent_id].discard(solution.id)
            if solution.generation in self._by_generation:
                self._by_generation[solution.generation].discard(solution.id)
            pruned += 1

        return pruned

    def save_to_json(self, path: str) -> None:
        """Save the tree to a JSON file."""
        data = {
            "max_generations": self.max_generations,
            "root_id": self.root_id,
            "current_generation": self.current_generation,
            "solutions": {
                sid: sol.to_dict()
                for sid, sol in self.solutions.items()
            },
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, path: str) -> "SolutionTree":
        """Load a tree from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tree = cls(max_generations=data.get("max_generations", 20))
        tree.root_id = data.get("root_id")
        tree.current_generation = data.get("current_generation", 0)

        # Load solutions
        for sid, sol_data in data.get("solutions", {}).items():
            solution = Solution.from_dict(sol_data)
            tree.solutions[sid] = solution

            # Rebuild indices
            if solution.parent_id:
                if solution.parent_id not in tree._children:
                    tree._children[solution.parent_id] = set()
                tree._children[solution.parent_id].add(solution.id)

            gen = solution.generation
            if gen not in tree._by_generation:
                tree._by_generation[gen] = set()
            tree._by_generation[gen].add(solution.id)

        return tree

    def __len__(self) -> int:
        return len(self.solutions)

    def __repr__(self) -> str:
        return (
            f"SolutionTree(solutions={len(self.solutions)}, "
            f"generations={self.current_generation + 1})"
        )
