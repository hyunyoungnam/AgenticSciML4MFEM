"""
Knowledge management for AgenticSciML.

Note: Static FEA knowledge base (KnowledgeBase, fea_knowledge.json) is deprecated.
Modern LLMs like Claude and GPT-4 have comprehensive FEA domain knowledge built-in.

FailureMemory is still used for runtime-specific learning from failed attempts.
"""

from inpforge.knowledge.failure_memory import FailureMemory, FailureEntry

# Deprecated - kept for backward compatibility
from inpforge.knowledge.base import KnowledgeEntry, KnowledgeBase

__all__ = [
    # Active components
    "FailureMemory",
    "FailureEntry",
    # Deprecated - LLMs have FEA knowledge built-in
    "KnowledgeEntry",
    "KnowledgeBase",
]
