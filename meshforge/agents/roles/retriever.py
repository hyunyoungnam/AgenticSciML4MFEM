"""
Retriever Agent implementation.

The Retriever searches the knowledge base for relevant FEA techniques
and expertise to guide mutation proposals.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from meshforge.agents.base import BaseAgent, AgentContext, AgentRole
from meshforge.knowledge.base import KnowledgeBase, KnowledgeEntry


# Retriever prompts
RETRIEVER_PROMPTS = {
    "system": """You are an FEA Knowledge Retriever Agent.

Your role is to:
1. Analyze queries about FEA modeling challenges
2. Identify relevant knowledge categories
3. Extract key search terms
4. Rank retrieved knowledge by relevance

You have expertise in:
- FEA terminology and concepts
- Mesh quality and refinement
- Material models and properties
- Solver convergence strategies
- Common failure modes

When given a query, identify:
- Primary topic area
- Related subtopics
- Key search terms
- Problem type indicators""",

    "analyze_query": """Analyze this query and suggest search strategy.

## Query
{query}

## Available Categories
{categories}

## Task
1. Identify primary category
2. Extract search keywords
3. Suggest related categories
4. Rate query specificity (high/medium/low)

Format:
**Primary Category**: [category]
**Keywords**: [keyword1, keyword2, ...]
**Related Categories**: [cat1, cat2, ...]
**Specificity**: [high/medium/low]""",
}


@dataclass
class RetrievalResult:
    """Result of knowledge retrieval."""
    entries: List[KnowledgeEntry] = field(default_factory=list)
    query: str = ""
    primary_category: str = ""
    keywords: List[str] = field(default_factory=list)
    total_found: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "query": self.query,
            "primary_category": self.primary_category,
            "keywords": self.keywords,
            "total_found": self.total_found,
        }

    def format_for_prompt(self, max_chars: int = 4000) -> str:
        """Format results for inclusion in agent prompts."""
        if not self.entries:
            return "No relevant knowledge found."

        lines = ["## Retrieved Knowledge"]
        current_chars = len(lines[0])

        for entry in self.entries:
            formatted = entry.format_for_prompt()
            if current_chars + len(formatted) + 10 > max_chars:
                break
            lines.append("")
            lines.append(formatted)
            current_chars += len(formatted) + 1

        return "\n".join(lines)


class RetrieverAgent(BaseAgent[RetrievalResult]):
    """
    Retriever Agent for knowledge base search.

    Responsibilities:
    1. Analyze queries to extract search terms
    2. Search knowledge base
    3. Rank and filter results
    4. Format knowledge for other agents
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        model: str = "gpt-4-turbo",
        temperature: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            role=AgentRole.RETRIEVER,
            model=model,
            temperature=temperature,
            **kwargs,
        )
        self.knowledge_base = knowledge_base or KnowledgeBase()

    def get_system_prompt(self) -> str:
        return RETRIEVER_PROMPTS["system"]

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        query = kwargs.get("query", "")
        categories = ", ".join(self.knowledge_base.CATEGORIES)

        return RETRIEVER_PROMPTS["analyze_query"].format(
            query=query,
            categories=categories,
        )

    def parse_response(self, response: str) -> RetrievalResult:
        """Parse LLM response - not used for direct retrieval."""
        return RetrievalResult()

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge entries.

        Args:
            query: Search query
            category: Optional category filter
            top_k: Maximum entries to return

        Returns:
            RetrievalResult
        """
        result = RetrievalResult(query=query)

        # Extract keywords from query
        result.keywords = self._extract_keywords(query)

        # Determine primary category
        if category:
            result.primary_category = category
        else:
            result.primary_category = self._infer_category(query)

        # Search
        entries = self.knowledge_base.search(
            query=query,
            category=result.primary_category if result.primary_category else None,
            top_k=top_k,
        )

        result.entries = entries
        result.total_found = len(entries)

        return result

    def retrieve_by_problem_type(self, problem_type: str) -> RetrievalResult:
        """
        Retrieve knowledge for a specific problem type.

        Args:
            problem_type: Problem type (e.g., "2D_hole", "plate_with_load")

        Returns:
            RetrievalResult
        """
        result = RetrievalResult(query=problem_type)

        entries = self.knowledge_base.search_by_problem_type(problem_type)
        result.entries = entries
        result.total_found = len(entries)

        return result

    def retrieve_for_error(self, error_type: str, error_message: str) -> RetrievalResult:
        """
        Retrieve knowledge relevant to an error.

        Args:
            error_type: Type of error
            error_message: Error message

        Returns:
            RetrievalResult
        """
        # Map error types to categories
        category_map = {
            "mesh_quality": "mesh_quality",
            "mesh": "mesh_quality",
            "convergence": "convergence",
            "material": "material",
            "boundary_condition": "boundary_conditions",
            "bc": "boundary_conditions",
        }

        category = category_map.get(error_type.lower(), None)

        # Build query from error
        query = f"{error_type} {error_message[:100]}"

        return self.retrieve(query=query, category=category, top_k=5)

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from a query."""
        import re

        # Remove common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "to", "of", "in", "for", "on", "with", "at",
            "by", "from", "as", "into", "through", "during", "before",
            "after", "above", "below", "between", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "and",
            "but", "if", "or", "because", "until", "while", "what",
        }

        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords[:10]  # Limit to 10 keywords

    def _infer_category(self, query: str) -> str:
        """Infer the most relevant category from a query."""
        query_lower = query.lower()

        category_keywords = {
            "mesh_quality": ["mesh", "element", "jacobian", "aspect ratio", "quality"],
            "material": ["material", "elastic", "modulus", "poisson", "steel", "aluminum"],
            "convergence": ["converge", "iteration", "solver", "cutback", "diverge"],
            "boundary_conditions": ["boundary", "bc", "constraint", "fixed", "load", "displacement"],
            "solver_settings": ["solver", "step", "increment", "nlgeom", "stabilization"],
            "element_types": ["element type", "cps", "cpe", "c3d", "quad", "triangle"],
            "best_practices": ["best practice", "recommend", "guideline", "avoid"],
            "common_errors": ["error", "warning", "fail", "problem", "issue"],
        }

        best_category = ""
        best_score = 0

        for category, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    def set_knowledge_base(self, knowledge_base: KnowledgeBase) -> None:
        """Set the knowledge base."""
        self.knowledge_base = knowledge_base

    async def execute(self, context: AgentContext, **kwargs) -> RetrievalResult:
        """
        Execute retrieval (direct search, no LLM needed).

        Args:
            context: Agent context
            **kwargs: query, category, top_k, etc.

        Returns:
            RetrievalResult
        """
        query = kwargs.get("query", "")
        category = kwargs.get("category")
        top_k = kwargs.get("top_k", 10)

        return self.retrieve(query=query, category=category, top_k=top_k)
