"""
Knowledge base for FEA domain expertise.

Provides structured storage and retrieval of curated FEA knowledge
entries that guide agent decision-making.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import json
from pathlib import Path
import re


@dataclass
class KnowledgeEntry:
    """
    A single knowledge entry in the FEA knowledge base.

    Represents a piece of domain knowledge about FEA modeling,
    mesh quality, convergence strategies, etc.

    Attributes:
        id: Unique identifier for the entry
        title: Short title describing the knowledge
        category: Category (mesh_quality, material, convergence, etc.)
        content: Main content/description
        keywords: Keywords for search
        applicable_to: Problem types this applies to
        severity: Importance level (info, warning, critical)
        examples: Example applications
        references: References to documentation
        metadata: Additional metadata
    """
    id: str
    title: str
    category: str
    content: str
    keywords: List[str] = field(default_factory=list)
    applicable_to: List[str] = field(default_factory=list)
    severity: str = "info"
    examples: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "content": self.content,
            "keywords": self.keywords,
            "applicable_to": self.applicable_to,
            "severity": self.severity,
            "examples": self.examples,
            "references": self.references,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        """Create entry from dictionary representation."""
        return cls(
            id=data["id"],
            title=data["title"],
            category=data["category"],
            content=data["content"],
            keywords=data.get("keywords", []),
            applicable_to=data.get("applicable_to", []),
            severity=data.get("severity", "info"),
            examples=data.get("examples", []),
            references=data.get("references", []),
            metadata=data.get("metadata", {}),
        )

    def matches_query(self, query: str) -> float:
        """
        Compute relevance score for a search query.

        Args:
            query: Search query string

        Returns:
            Relevance score (0-1)
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        score = 0.0

        # Title match (highest weight)
        if query_lower in self.title.lower():
            score += 0.4
        else:
            title_words = set(re.findall(r'\w+', self.title.lower()))
            title_overlap = len(query_words & title_words) / max(len(query_words), 1)
            score += 0.3 * title_overlap

        # Keyword match
        keyword_lower = [k.lower() for k in self.keywords]
        for keyword in keyword_lower:
            if keyword in query_lower or query_lower in keyword:
                score += 0.15
                break

        # Content match
        if query_lower in self.content.lower():
            score += 0.2
        else:
            content_words = set(re.findall(r'\w+', self.content.lower()))
            content_overlap = len(query_words & content_words) / max(len(query_words), 1)
            score += 0.15 * content_overlap

        # Category match
        if query_lower in self.category.lower():
            score += 0.1

        return min(1.0, score)

    def format_for_prompt(self) -> str:
        """Format the entry for inclusion in an LLM prompt."""
        lines = [
            f"### {self.title}",
            f"Category: {self.category}",
            f"Severity: {self.severity}",
            "",
            self.content,
        ]

        if self.examples:
            lines.append("")
            lines.append("Examples:")
            for example in self.examples:
                lines.append(f"- {example}")

        if self.keywords:
            lines.append("")
            lines.append(f"Keywords: {', '.join(self.keywords)}")

        return "\n".join(lines)


class KnowledgeBase:
    """
    Knowledge base for FEA domain expertise.

    Stores and retrieves KnowledgeEntry objects for guiding
    agent decision-making during mutation proposal and validation.
    """

    # Standard categories
    CATEGORIES = [
        "mesh_quality",
        "material",
        "convergence",
        "boundary_conditions",
        "solver_settings",
        "element_types",
        "contact",
        "plasticity",
        "dynamics",
        "thermal",
        "best_practices",
        "common_errors",
    ]

    def __init__(self):
        """Initialize an empty knowledge base."""
        self.entries: Dict[str, KnowledgeEntry] = {}
        self._index_by_category: Dict[str, Set[str]] = {cat: set() for cat in self.CATEGORIES}
        self._index_by_keyword: Dict[str, Set[str]] = {}

    def add_entry(self, entry: KnowledgeEntry) -> None:
        """
        Add an entry to the knowledge base.

        Args:
            entry: KnowledgeEntry to add
        """
        self.entries[entry.id] = entry

        # Index by category
        if entry.category in self._index_by_category:
            self._index_by_category[entry.category].add(entry.id)
        else:
            self._index_by_category[entry.category] = {entry.id}

        # Index by keywords
        for keyword in entry.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self._index_by_keyword:
                self._index_by_keyword[keyword_lower] = set()
            self._index_by_keyword[keyword_lower].add(entry.id)

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get an entry by ID."""
        return self.entries.get(entry_id)

    def get_by_category(self, category: str) -> List[KnowledgeEntry]:
        """Get all entries in a category."""
        entry_ids = self._index_by_category.get(category, set())
        return [self.entries[eid] for eid in entry_ids if eid in self.entries]

    def get_by_keyword(self, keyword: str) -> List[KnowledgeEntry]:
        """Get all entries with a specific keyword."""
        keyword_lower = keyword.lower()
        entry_ids = self._index_by_keyword.get(keyword_lower, set())
        return [self.entries[eid] for eid in entry_ids if eid in self.entries]

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.1,
    ) -> List[KnowledgeEntry]:
        """
        Search the knowledge base.

        Args:
            query: Search query string
            category: Optional category filter
            top_k: Maximum number of results
            min_score: Minimum relevance score

        Returns:
            List of matching entries, sorted by relevance
        """
        candidates = []

        # Get candidate entries
        if category:
            entry_ids = self._index_by_category.get(category, set())
            entries = [self.entries[eid] for eid in entry_ids if eid in self.entries]
        else:
            entries = list(self.entries.values())

        # Score and filter
        for entry in entries:
            score = entry.matches_query(query)
            if score >= min_score:
                candidates.append((entry, score))

        # Sort by score and return top k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, score in candidates[:top_k]]

    def search_by_problem_type(self, problem_type: str) -> List[KnowledgeEntry]:
        """
        Search for entries applicable to a specific problem type.

        Args:
            problem_type: Problem type (e.g., "2D_hole", "plate_with_load")

        Returns:
            List of applicable entries
        """
        results = []
        for entry in self.entries.values():
            if problem_type in entry.applicable_to or "all" in entry.applicable_to:
                results.append(entry)
        return results

    def load_from_json(self, json_path: str) -> int:
        """
        Load entries from a JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            Number of entries loaded
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {json_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        entries_data = data if isinstance(data, list) else data.get("entries", [])

        for entry_data in entries_data:
            entry = KnowledgeEntry.from_dict(entry_data)
            self.add_entry(entry)
            count += 1

        return count

    def save_to_json(self, json_path: str) -> None:
        """
        Save entries to a JSON file.

        Args:
            json_path: Path to JSON file
        """
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "entries": [entry.to_dict() for entry in self.entries.values()]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def format_context(
        self,
        entries: List[KnowledgeEntry],
        max_chars: int = 4000,
    ) -> str:
        """
        Format a list of entries for inclusion in an LLM context.

        Args:
            entries: List of entries to format
            max_chars: Maximum character limit

        Returns:
            Formatted context string
        """
        lines = ["## Relevant FEA Knowledge\n"]

        current_chars = len(lines[0])
        for entry in entries:
            formatted = entry.format_for_prompt()
            if current_chars + len(formatted) + 10 > max_chars:
                break
            lines.append(formatted)
            lines.append("")
            current_chars += len(formatted) + 1

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        category_counts = {
            cat: len(ids) for cat, ids in self._index_by_category.items()
        }

        return {
            "total_entries": len(self.entries),
            "categories": category_counts,
            "total_keywords": len(self._index_by_keyword),
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"KnowledgeBase(entries={len(self.entries)})"
