"""
Knowledge management for FEA domain expertise.

Provides curated FEA knowledge entries and dynamic failure memory
to guide agent decision-making.
"""

from knowledge.base import KnowledgeEntry, KnowledgeBase
from knowledge.failure_memory import FailureMemory, FailureEntry

__all__ = [
    "KnowledgeEntry",
    "KnowledgeBase",
    "FailureMemory",
    "FailureEntry",
]
