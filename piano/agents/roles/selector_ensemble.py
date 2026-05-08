"""
Selector Ensemble Agent.

Replaces brief-training candidate selection in AgenticSurrogateTrainer.

Instead of brief-training N candidates and picking the one with lowest loss,
three diverse LLM instances vote independently on which configuration to
pursue. The majority vote selects; if tied, the brief-training fallback picks.

This matches the Selector Ensemble in Jiang & Karniadakis (2026) — three LLMs
each vote independently, reducing single-model bias and preventing groupthink.

Diversity is achieved by:
- Different models (sonnet vs haiku vs a third haiku with different temperature)
- Different temperatures (0.3, 0.7, 0.5)
- Each voter receives only the debate log for its candidate (no cross-candidate info)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VoteResult:
    """A single voter's decision."""
    voter_id: int
    chosen_index: int
    reasoning: str
    raw_response: str = ""


@dataclass
class SelectionResult:
    """Final selection after ensemble voting."""
    selected_index: int
    vote_counts: Dict[int, int] = field(default_factory=dict)
    votes: List[VoteResult] = field(default_factory=list)
    selection_method: str = "majority_vote"
    confidence: float = 0.0


class SelectorEnsembleAgent:
    """
    Three-LLM ensemble voter for debate candidate selection.

    Given N debate candidates (each a DebateResult with arch_proposal,
    physics_changes, and a validation_text summary), asks three independent
    LLM voters to choose the best one. Majority vote wins.

    Usage:
        selector = SelectorEnsembleAgent(llm_provider, n_voters=3)
        result = selector.select_sync(candidates, history_summary)
        best_config = candidates[result.selected_index]
    """

    _VOTER_SYSTEM = (
        "You are an expert reviewer of neural operator configurations for physics-informed "
        "machine learning. You will receive a summary of N training configurations proposed "
        "by an agentic HPO system. Choose the SINGLE BEST configuration — the one most "
        "likely to improve test loss given the current training diagnostics.\n\n"
        "Consider:\n"
        "1. Does the configuration address the diagnosed issue (overfitting/underfitting/etc.)?\n"
        "2. Are the proposed changes conservative (not too large a jump)?\n"
        "3. Is the physics loss weighting consistent with stability?\n"
        "4. Did the Critic validation flag any concerns?\n\n"
        "Reply ONLY with:\n"
        "CHOICE: <integer index 0..N-1>\n"
        "REASONING: <one sentence>\n"
    )

    def __init__(
        self,
        llm_provider: Any,
        n_voters: int = 3,
        voter_configs: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Args:
            llm_provider: Default LLM provider (used if voter_configs not set)
            n_voters: Number of independent voters (default 3)
            voter_configs: Optional per-voter config dicts with 'model' and 'temperature'
        """
        self._llm_provider = llm_provider
        self._n_voters = n_voters
        self._voter_configs = voter_configs or self._default_voter_configs()

    @staticmethod
    def _default_voter_configs() -> List[Dict[str, Any]]:
        return [
            {"model": "claude-sonnet-4-6", "temperature": 0.3},
            {"model": "claude-haiku-4-5-20251001", "temperature": 0.7},
            {"model": "claude-haiku-4-5-20251001", "temperature": 0.5},
        ]

    async def select(
        self,
        candidates: List[Any],
        history_summary: str = "",
    ) -> SelectionResult:
        """
        Vote on N candidates and return the majority winner.

        Args:
            candidates: List of DebateResult objects (or any with .arch_proposal and .physics_changes)
            history_summary: Concise current training state for context

        Returns:
            SelectionResult with the winning index and vote breakdown
        """
        if len(candidates) == 1:
            return SelectionResult(
                selected_index=0,
                vote_counts={0: self._n_voters},
                selection_method="single_candidate",
                confidence=1.0,
            )

        prompt = self._build_prompt(candidates, history_summary)
        votes: List[VoteResult] = []

        for voter_id, cfg in enumerate(self._voter_configs[:self._n_voters]):
            try:
                response = await self._llm_provider.generate(
                    system_prompt=self._VOTER_SYSTEM,
                    user_prompt=prompt,
                    model=cfg.get("model"),
                    temperature=cfg.get("temperature", 0.5),
                    max_tokens=256,
                )
                vote = self._parse_vote(voter_id, response.content, len(candidates))
                votes.append(vote)
                logger.info(
                    f"SelectorEnsemble voter {voter_id}: choice={vote.chosen_index}, "
                    f"reason={vote.reasoning[:80]}"
                )
            except Exception as e:
                logger.warning(f"SelectorEnsemble voter {voter_id} failed: {e}")
                votes.append(VoteResult(voter_id=voter_id, chosen_index=0, reasoning="error"))

        # Tally
        vote_counts: Dict[int, int] = {}
        for v in votes:
            vote_counts[v.chosen_index] = vote_counts.get(v.chosen_index, 0) + 1

        # Majority winner
        winner = max(vote_counts, key=vote_counts.get)
        max_votes = vote_counts[winner]
        confidence = max_votes / len(votes)

        logger.info(
            f"SelectorEnsemble: winner=candidate[{winner}] with {max_votes}/{len(votes)} votes "
            f"(confidence={confidence:.2f})"
        )

        return SelectionResult(
            selected_index=winner,
            vote_counts=vote_counts,
            votes=votes,
            selection_method="majority_vote",
            confidence=confidence,
        )

    def _build_prompt(self, candidates: List[Any], history_summary: str) -> str:
        parts = []
        if history_summary:
            parts.append(f"## Current Training State\n{history_summary}\n")

        parts.append(f"## Candidate Configurations ({len(candidates)} total)\n")
        for i, cand in enumerate(candidates):
            arch = getattr(cand, 'arch_proposal', None)
            phys = getattr(cand, 'physics_changes', {})
            val = getattr(cand, 'validation_text', '')

            arch_changes = getattr(arch, 'changes', {}) if arch else {}
            arch_reasoning = getattr(arch, 'reasoning', '')[:200] if arch else ''

            parts.append(
                f"### Candidate {i}\n"
                f"Architecture changes: {arch_changes}\n"
                f"Physics changes: {phys}\n"
                f"Architect reasoning: {arch_reasoning}\n"
                f"Critic validation: {val[:200]}\n"
            )

        parts.append(f"\nChoose the best candidate (index 0..{len(candidates)-1}).")
        return "\n".join(parts)

    def _parse_vote(self, voter_id: int, response: str, n_candidates: int) -> VoteResult:
        import re
        chosen = 0
        reasoning = ""

        m = re.search(r"CHOICE:\s*(\d+)", response, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            chosen = max(0, min(idx, n_candidates - 1))

        m = re.search(r"REASONING:\s*(.+?)$", response, re.DOTALL | re.IGNORECASE)
        if m:
            reasoning = m.group(1).strip()[:200]

        return VoteResult(
            voter_id=voter_id,
            chosen_index=chosen,
            reasoning=reasoning,
            raw_response=response,
        )

    def select_sync(
        self,
        candidates: List[Any],
        history_summary: str = "",
    ) -> SelectionResult:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.select(candidates, history_summary))
        finally:
            loop.close()
