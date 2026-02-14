"""
Failure memory for tracking and learning from past failures.

Maintains a dynamic log of failures encountered during evolution,
enabling agents to avoid repeating mistakes.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FailureEntry:
    """A single failure entry in the memory."""
    id: str
    solution_id: str
    delta_R: Optional[float]
    error_type: str
    error_message: str
    generation: int
    timestamp: datetime = field(default_factory=datetime.now)
    mutation_params: Dict[str, Any] = field(default_factory=dict)
    attempted_fixes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "solution_id": self.solution_id,
            "delta_R": self.delta_R,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "generation": self.generation,
            "timestamp": self.timestamp.isoformat(),
            "mutation_params": self.mutation_params,
            "attempted_fixes": self.attempted_fixes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureEntry":
        entry = cls(
            id=data["id"],
            solution_id=data["solution_id"],
            delta_R=data.get("delta_R"),
            error_type=data.get("error_type", "unknown"),
            error_message=data.get("error_message", ""),
            generation=data.get("generation", 0),
            mutation_params=data.get("mutation_params", {}),
            attempted_fixes=data.get("attempted_fixes", []),
            metadata=data.get("metadata", {}),
        )
        if data.get("timestamp"):
            entry.timestamp = datetime.fromisoformat(data["timestamp"])
        return entry

    def matches_pattern(self, delta_R: Optional[float] = None, tolerance: float = 0.1) -> bool:
        """Check if this failure matches a given pattern."""
        if delta_R is not None and self.delta_R is not None:
            if abs(delta_R - self.delta_R) < tolerance:
                return True
        return False


class FailureMemory:
    """
    Dynamic failure memory for the AgenticSciML system.

    Tracks failures and provides:
    - Failure pattern matching
    - Safe parameter range estimation
    - Failure statistics by type
    """

    def __init__(self, max_entries: int = 1000):
        """
        Initialize failure memory.

        Args:
            max_entries: Maximum number of entries to retain
        """
        self.max_entries = max_entries
        self.entries: Dict[str, FailureEntry] = {}
        self._by_error_type: Dict[str, List[str]] = {}
        self._by_generation: Dict[int, List[str]] = {}
        self._delta_R_failures: List[float] = []
        self._entry_counter = 0

    def add_failure(
        self,
        solution_id: str,
        delta_R: Optional[float],
        error: str,
        generation: int,
        error_type: str = "unknown",
        mutation_params: Optional[Dict[str, Any]] = None,
    ) -> FailureEntry:
        """
        Add a failure to memory.

        Args:
            solution_id: ID of the failed solution
            delta_R: Delta R value that caused failure
            error: Error message
            generation: Generation number
            error_type: Type of error
            mutation_params: Mutation parameters

        Returns:
            Created FailureEntry
        """
        self._entry_counter += 1
        entry_id = f"failure_{self._entry_counter}"

        entry = FailureEntry(
            id=entry_id,
            solution_id=solution_id,
            delta_R=delta_R,
            error_type=self._classify_error(error) if error_type == "unknown" else error_type,
            error_message=error,
            generation=generation,
            mutation_params=mutation_params or {},
        )

        self.entries[entry_id] = entry

        # Index by error type
        if entry.error_type not in self._by_error_type:
            self._by_error_type[entry.error_type] = []
        self._by_error_type[entry.error_type].append(entry_id)

        # Index by generation
        if generation not in self._by_generation:
            self._by_generation[generation] = []
        self._by_generation[generation].append(entry_id)

        # Track delta_R failures
        if delta_R is not None:
            self._delta_R_failures.append(delta_R)

        # Prune if needed
        if len(self.entries) > self.max_entries:
            self._prune_oldest(len(self.entries) - self.max_entries)

        return entry

    def _classify_error(self, error: str) -> str:
        """Classify an error message into a type."""
        error_lower = error.lower()

        if any(x in error_lower for x in ["jacobian", "distort", "mesh"]):
            return "mesh_quality"
        elif any(x in error_lower for x in ["converge", "iteration", "cutback"]):
            return "convergence"
        elif any(x in error_lower for x in ["material", "elastic"]):
            return "material"
        elif any(x in error_lower for x in ["boundary", "bc", "constraint"]):
            return "boundary_condition"
        elif any(x in error_lower for x in ["file", "path", "import"]):
            return "file_error"
        else:
            return "other"

    def get_recent_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent failures."""
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.timestamp,
            reverse=True
        )
        return [e.to_dict() for e in sorted_entries[:limit]]

    def get_failures_by_type(self, error_type: str) -> List[FailureEntry]:
        """Get all failures of a specific type."""
        entry_ids = self._by_error_type.get(error_type, [])
        return [self.entries[eid] for eid in entry_ids if eid in self.entries]

    def get_failures_by_generation(self, generation: int) -> List[FailureEntry]:
        """Get all failures in a specific generation."""
        entry_ids = self._by_generation.get(generation, [])
        return [self.entries[eid] for eid in entry_ids if eid in self.entries]

    def get_safe_delta_R_range(self, margin: float = 0.2) -> tuple:
        """
        Estimate safe delta_R range based on failure history.

        Args:
            margin: Safety margin below minimum failure

        Returns:
            Tuple of (min_safe, max_safe) delta_R values
        """
        if not self._delta_R_failures:
            return (-1.0, 2.0)  # Default range

        failed_values = self._delta_R_failures
        min_failed = min(failed_values)
        max_failed = max(failed_values)

        # Estimate safe range
        if min_failed > 0:
            # All failures at positive delta_R
            safe_max = min_failed - margin
            safe_min = -1.0
        elif max_failed < 0:
            # All failures at negative delta_R
            safe_min = max_failed + margin
            safe_max = 2.0
        else:
            # Failures on both sides
            safe_min = -0.5
            safe_max = 1.0

        return (safe_min, safe_max)

    def has_similar_failure(
        self,
        delta_R: Optional[float] = None,
        tolerance: float = 0.1,
    ) -> bool:
        """
        Check if a similar failure has occurred.

        Args:
            delta_R: Delta R value to check
            tolerance: Matching tolerance

        Returns:
            True if similar failure exists
        """
        for entry in self.entries.values():
            if entry.matches_pattern(delta_R, tolerance):
                return True
        return False

    def get_failure_count_by_type(self) -> Dict[str, int]:
        """Get failure counts by error type."""
        return {
            error_type: len(entry_ids)
            for error_type, entry_ids in self._by_error_type.items()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall failure statistics."""
        return {
            "total_failures": len(self.entries),
            "by_type": self.get_failure_count_by_type(),
            "by_generation": {
                gen: len(ids) for gen, ids in self._by_generation.items()
            },
            "safe_delta_R_range": self.get_safe_delta_R_range(),
            "delta_R_failures": len(self._delta_R_failures),
        }

    def format_for_prompt(self, limit: int = 5) -> str:
        """Format recent failures for inclusion in agent prompts."""
        recent = self.get_recent_failures(limit)

        if not recent:
            return "No previous failures recorded."

        lines = ["## Previous Failures"]
        for i, failure in enumerate(recent, 1):
            lines.append(f"\n### Failure {i}")
            lines.append(f"- delta_R: {failure.get('delta_R', 'N/A')}")
            lines.append(f"- Error: {failure.get('error_message', 'Unknown')[:200]}")
            lines.append(f"- Type: {failure.get('error_type', 'unknown')}")

        return "\n".join(lines)

    def _prune_oldest(self, count: int) -> None:
        """Prune oldest entries."""
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda x: x[1].timestamp
        )

        for entry_id, entry in sorted_entries[:count]:
            del self.entries[entry_id]

            # Clean up indices
            if entry.error_type in self._by_error_type:
                if entry_id in self._by_error_type[entry.error_type]:
                    self._by_error_type[entry.error_type].remove(entry_id)

            if entry.generation in self._by_generation:
                if entry_id in self._by_generation[entry.generation]:
                    self._by_generation[entry.generation].remove(entry_id)

    def save_to_json(self, path: str) -> None:
        """Save failure memory to JSON file."""
        data = {
            "entries": [e.to_dict() for e in self.entries.values()],
            "statistics": self.get_statistics(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, path: str) -> "FailureMemory":
        """Load failure memory from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        memory = cls()

        for entry_data in data.get("entries", []):
            entry = FailureEntry.from_dict(entry_data)
            memory.entries[entry.id] = entry

            # Rebuild indices
            if entry.error_type not in memory._by_error_type:
                memory._by_error_type[entry.error_type] = []
            memory._by_error_type[entry.error_type].append(entry.id)

            if entry.generation not in memory._by_generation:
                memory._by_generation[entry.generation] = []
            memory._by_generation[entry.generation].append(entry.id)

            if entry.delta_R is not None:
                memory._delta_R_failures.append(entry.delta_R)

        return memory

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"FailureMemory(entries={len(self.entries)})"
