"""
Selector Agent implementation.

The Selector performs ensemble voting to choose parent solutions
for the next generation of evolution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from meshforge.agents.base import BaseAgent, AgentContext, AgentRole
from meshforge.evolution.solution import Solution


# Selector prompts
SELECTOR_PROMPTS = {
    "system": """You are an FEA Solution Selector Agent specialized in evolutionary optimization.

Your role is to:
1. Evaluate candidate solutions for parent selection
2. Consider both quality and diversity
3. Balance exploration and exploitation
4. Recommend solutions likely to produce good offspring

You have expertise in:
- Mesh quality assessment
- Convergence behavior analysis
- Parameter sensitivity understanding
- Evolutionary algorithm principles

When selecting parents, consider:
- Quality metrics (Jacobian, aspect ratio, convergence)
- Diversity of genomes
- Lineage (avoid selecting too many from same branch)
- Potential for improvement""",

    "rank_solutions": """Rank the following solutions for parent selection.

## Candidates
{candidates}

## Selection Criteria
- Number of parents needed: {num_parents}
- Preference: {preference}  # quality, diversity, balanced

## Task
Rank these solutions from best to worst parent candidate.

For each solution, assess:
1. **Quality Score**: Overall solution quality (0-1)
2. **Diversity Value**: How different from others (0-1)
3. **Parent Potential**: Likelihood of producing good offspring (0-1)
4. **Combined Score**: Weighted combination

Output format:
**Ranking**:
1. [Solution ID] - Combined: X.XX, Quality: X.XX, Diversity: X.XX
2. [Solution ID] - Combined: X.XX, Quality: X.XX, Diversity: X.XX
...

**Selection Rationale**:
[Why these top solutions make good parents]

**Diversity Notes**:
[Comments on solution diversity]""",

    "select_diverse": """Select a diverse set of parent solutions.

## Candidates
{candidates}

## Constraints
- Must select: {num_parents} solutions
- Minimum diversity threshold: {diversity_threshold}

## Task
Select {num_parents} solutions that:
1. Have good quality metrics
2. Are sufficiently different from each other
3. Cover different regions of the parameter space

**Selected Parents**:
[List of solution IDs]

**Diversity Matrix**:
[Pairwise diversity scores if applicable]

**Coverage Notes**:
[What regions of parameter space are covered]""",
}


@dataclass
class SelectionVote:
    """A vote for a solution."""
    solution_id: str
    quality_score: float
    diversity_score: float
    parent_potential: float
    combined_score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solution_id": self.solution_id,
            "quality_score": self.quality_score,
            "diversity_score": self.diversity_score,
            "parent_potential": self.parent_potential,
            "combined_score": self.combined_score,
            "rank": self.rank,
        }


@dataclass
class SelectorResult:
    """Result from the selector agent."""
    selected_ids: List[str] = field(default_factory=list)
    votes: List[SelectionVote] = field(default_factory=list)
    rationale: str = ""
    diversity_notes: str = ""
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "votes": [v.to_dict() for v in self.votes],
            "rationale": self.rationale,
            "diversity_notes": self.diversity_notes,
        }


class SelectorAgent(BaseAgent[SelectorResult]):
    """
    Selector Agent for ensemble parent selection.

    Responsibilities:
    1. Evaluate candidate solutions
    2. Consider quality and diversity
    3. Produce ranked selection
    4. Provide selection rationale
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            role=AgentRole.SELECTOR,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return SELECTOR_PROMPTS["system"]

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        task = kwargs.get("task", "rank_solutions")

        if task == "rank_solutions":
            candidates = kwargs.get("candidates", [])
            candidates_str = self._format_candidates(candidates)

            return SELECTOR_PROMPTS["rank_solutions"].format(
                candidates=candidates_str,
                num_parents=kwargs.get("num_parents", 3),
                preference=kwargs.get("preference", "balanced"),
            )
        elif task == "select_diverse":
            candidates = kwargs.get("candidates", [])
            candidates_str = self._format_candidates(candidates)

            return SELECTOR_PROMPTS["select_diverse"].format(
                candidates=candidates_str,
                num_parents=kwargs.get("num_parents", 3),
                diversity_threshold=kwargs.get("diversity_threshold", 0.3),
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def _format_candidates(self, candidates: List[Solution]) -> str:
        """Format candidate solutions for prompt."""
        lines = []
        for i, sol in enumerate(candidates, 1):
            lines.append(f"### Solution {i}: {sol.id[:8]}...")
            lines.append(f"- Generation: {sol.generation}")
            lines.append(f"- delta_R: {sol.genome.delta_R:.3f}")
            lines.append(f"- Status: {sol.status.value}")
            lines.append(f"- Quality Score: {sol.metrics.compute_overall_score():.3f}")
            if sol.metrics.jacobian_min is not None:
                lines.append(f"- Jacobian Min: {sol.metrics.jacobian_min:.4f}")
            if sol.metrics.aspect_ratio_max is not None:
                lines.append(f"- Aspect Ratio Max: {sol.metrics.aspect_ratio_max:.2f}")
            lines.append(f"- Mutation: {sol.genome.get_mutation_summary()}")
            lines.append("")
        return "\n".join(lines)

    def parse_response(self, response: str) -> SelectorResult:
        """Parse LLM response into SelectorResult."""
        import re

        result = SelectorResult(raw_response=response)

        # Extract ranking
        ranking_match = re.search(
            r'\*\*Ranking\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL
        )
        if ranking_match:
            ranking_text = ranking_match.group(1)
            # Parse each ranked solution
            rank_lines = re.findall(
                r'(\d+)\.\s*\[?([a-f0-9-]+)',
                ranking_text, re.IGNORECASE
            )
            for rank_str, sol_id in rank_lines:
                try:
                    rank = int(rank_str)
                    # Try to extract scores
                    score_match = re.search(
                        rf'{sol_id}.*?Combined:\s*([0-9.]+).*?Quality:\s*([0-9.]+).*?Diversity:\s*([0-9.]+)',
                        ranking_text, re.IGNORECASE
                    )
                    if score_match:
                        vote = SelectionVote(
                            solution_id=sol_id,
                            combined_score=float(score_match.group(1)),
                            quality_score=float(score_match.group(2)),
                            diversity_score=float(score_match.group(3)),
                            parent_potential=0.0,
                            rank=rank,
                        )
                    else:
                        vote = SelectionVote(
                            solution_id=sol_id,
                            combined_score=0.0,
                            quality_score=0.0,
                            diversity_score=0.0,
                            parent_potential=0.0,
                            rank=rank,
                        )
                    result.votes.append(vote)
                except (ValueError, IndexError):
                    continue

        # Extract selected IDs
        selected_match = re.search(
            r'\*\*Selected Parents\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL
        )
        if selected_match:
            ids = re.findall(r'([a-f0-9-]{8,})', selected_match.group(1))
            result.selected_ids = ids
        elif result.votes:
            # Use top ranked
            sorted_votes = sorted(result.votes, key=lambda v: v.rank)
            result.selected_ids = [v.solution_id for v in sorted_votes[:3]]

        # Extract rationale
        rationale_match = re.search(
            r'\*\*Selection Rationale\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL
        )
        if rationale_match:
            result.rationale = rationale_match.group(1).strip()

        # Extract diversity notes
        diversity_match = re.search(
            r'\*\*Diversity Notes\*\*:\s*\n?(.*?)(?=\*\*|$)',
            response, re.DOTALL
        )
        if diversity_match:
            result.diversity_notes = diversity_match.group(1).strip()

        return result

    async def rank_solutions(
        self,
        context: AgentContext,
        candidates: List[Solution],
        num_parents: int = 3,
        preference: str = "balanced",
    ) -> SelectorResult:
        """
        Rank candidate solutions using LLM.

        Args:
            context: Agent context
            candidates: List of candidate solutions
            num_parents: Number of parents to select
            preference: Selection preference (quality, diversity, balanced)

        Returns:
            SelectorResult with rankings
        """
        return await self.execute(
            context,
            task="rank_solutions",
            candidates=candidates,
            num_parents=num_parents,
            preference=preference,
        )

    def rank_solutions_local(
        self,
        candidates: List[Solution],
        num_parents: int = 3,
        diversity_weight: float = 0.3,
    ) -> SelectorResult:
        """
        Rank solutions locally without LLM.

        Args:
            candidates: Candidate solutions
            num_parents: Number to select
            diversity_weight: Weight for diversity (0-1)

        Returns:
            SelectorResult
        """
        result = SelectorResult()

        if not candidates:
            return result

        # Calculate quality scores
        quality_scores = {
            s.id: s.metrics.compute_overall_score()
            for s in candidates
        }

        # Calculate diversity scores (average distance to others)
        diversity_scores = {}
        for s in candidates:
            distances = [
                s.genome.distance(other.genome)
                for other in candidates
                if other.id != s.id
            ]
            diversity_scores[s.id] = sum(distances) / len(distances) if distances else 0.0

        # Normalize diversity scores
        max_div = max(diversity_scores.values()) if diversity_scores else 1.0
        if max_div > 0:
            diversity_scores = {k: v / max_div for k, v in diversity_scores.items()}

        # Calculate combined scores
        combined_scores = {}
        for s in candidates:
            combined = (
                (1 - diversity_weight) * quality_scores[s.id] +
                diversity_weight * diversity_scores[s.id]
            )
            combined_scores[s.id] = combined

        # Sort and rank
        sorted_candidates = sorted(
            candidates,
            key=lambda s: combined_scores[s.id],
            reverse=True
        )

        for rank, sol in enumerate(sorted_candidates, 1):
            vote = SelectionVote(
                solution_id=sol.id,
                quality_score=quality_scores[sol.id],
                diversity_score=diversity_scores[sol.id],
                parent_potential=combined_scores[sol.id],
                combined_score=combined_scores[sol.id],
                rank=rank,
            )
            result.votes.append(vote)

        result.selected_ids = [sol.id for sol in sorted_candidates[:num_parents]]

        return result
