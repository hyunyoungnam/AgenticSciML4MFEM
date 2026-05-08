"""
Knowledge Retriever Agent.

Surfaces relevant SciML technique entries from the knowledge base before each
HPO debate round. Based on the parent node's weaknesses (Critic diagnosis +
Analyst observation), retrieves 0-1 KB entries whose 'apply_when' condition
matches the current training state.

Ablation study (Jiang & Karniadakis 2026): KB retrieval gives 2.3x-20x
improvement over no-KB and random-KB baselines respectively.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_KB_PATH = Path(__file__).parents[3] / "knowledge_base" / "kb_index.json"


@dataclass
class KBEntry:
    """A single knowledge base entry."""
    method_name: str
    category: str
    description: str
    keywords: List[str]
    filepath: str
    apply_when: str
    content: str = ""

    def to_context_string(self) -> str:
        return (
            f"[KNOWLEDGE BASE — {self.method_name}]\n"
            f"Category: {self.category}\n"
            f"Description: {self.description}\n"
            f"Apply when: {self.apply_when}\n\n"
            f"{self.content}"
        )


class KnowledgeRetrieverAgent:
    """
    Retrieves relevant KB entries based on current training weaknesses.

    Algorithm:
    1. Build a query string from Critic diagnosis + Analyst pattern
    2. Score each KB entry by keyword overlap with the query
    3. Return the single highest-scoring entry if score > threshold; else None

    The returned entry is injected into AgentContext.knowledge_context before
    Rounds 1-3 of the debate so all agents receive domain knowledge.

    Usage:
        retriever = KnowledgeRetrieverAgent()
        entry = retriever.retrieve(
            diagnosis="OVERFITTING",
            analyst_pattern="near-tip error high, stress_intensity not converging",
            pino_status="stress_intensity plateaued at 0.03",
        )
        if entry:
            context.knowledge_context = [{"text": entry.to_context_string()}]
    """

    def __init__(self, kb_path: Optional[Path] = None):
        self._kb_path = Path(kb_path) if kb_path else _DEFAULT_KB_PATH
        self._entries: List[KBEntry] = []
        self._load()

    def _load(self) -> None:
        """Load KB index and entry content from disk."""
        if not self._kb_path.exists():
            logger.warning(f"KB index not found at {self._kb_path}; knowledge retrieval disabled")
            return

        with open(self._kb_path) as f:
            index = json.load(f)

        kb_dir = self._kb_path.parent
        for rec in index:
            entry = KBEntry(**rec)
            content_path = kb_dir / Path(rec["filepath"]).name
            if content_path.exists():
                entry.content = content_path.read_text()
            else:
                entry.content = f"(content not found at {content_path})"
            self._entries.append(entry)

        logger.info(f"KnowledgeRetriever: loaded {len(self._entries)} entries from {self._kb_path}")

    def retrieve(
        self,
        diagnosis: str = "",
        analyst_pattern: str = "",
        pino_status: str = "",
        config_history: Optional[List[Dict]] = None,
        score_threshold: float = 1.0,
    ) -> Optional[KBEntry]:
        """
        Return the single most relevant KB entry, or None if no match.

        Args:
            diagnosis: Primary training issue from HyperparameterCriticAgent
              (e.g. "OVERFITTING", "UNDERFITTING", "LOSS_PLATEAU")
            analyst_pattern: Free-text pattern from ResultAnalystAgent
              (e.g. "near-tip error high, stress_intensity plateaued")
            pino_status: PINO term status from AnalystObservation
            config_history: Previous HPO rounds (to avoid re-suggesting same technique)
            score_threshold: Minimum keyword overlap score to return an entry
        """
        if not self._entries:
            return None

        query = " ".join([
            diagnosis.lower(),
            analyst_pattern.lower(),
            pino_status.lower(),
        ])

        # Techniques already used in config history
        used_methods = set()
        if config_history:
            for h in config_history:
                used_methods.update(
                    m.strip()
                    for m in re.findall(r'kb_technique:\s*(.+?)(?:\n|$)', str(h.get("changes", "")))
                )

        best_entry: Optional[KBEntry] = None
        best_score: float = 0.0

        for entry in self._entries:
            if entry.method_name in used_methods:
                continue

            score = sum(1.0 for kw in entry.keywords if kw.lower() in query)

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score < score_threshold:
            logger.debug(f"KnowledgeRetriever: no entry matched (best_score={best_score:.1f})")
            return None

        logger.info(
            f"KnowledgeRetriever: retrieved '{best_entry.method_name}' "
            f"(score={best_score:.1f}, category={best_entry.category})"
        )
        return best_entry

    def retrieve_by_name(self, method_name: str) -> Optional[KBEntry]:
        """Direct lookup by method name (for testing)."""
        for entry in self._entries:
            if entry.method_name.lower() == method_name.lower():
                return entry
        return None

    @property
    def n_entries(self) -> int:
        return len(self._entries)
